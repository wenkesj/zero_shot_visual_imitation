# -*- coding: utf-8 -*-
from __future__ import print_function

from alchemy.envs import VizDoomEnv, ResolutionWrapper
from alchemy.memory import FIFOMemory, rollout_dataset, RolloutPool, rollout
from alchemy.models import VisualImitationModel
from alchemy.spaces import CategoricalSpace, ContinuousSpace
from alchemy.tools import DemoRecorder

from matplotlib import pyplot as plt
from matplotlib import animation

import math

import numpy as np

import os

import tensorflow as tf

import threading

from time import sleep

from tqdm import trange


batch_size = 8
num_envs = 2
num_episodes = int(math.ceil(batch_size / num_envs))
max_trajectory_steps = max_sequence_length = 16
max_rollout_steps = 512


def action_value_reconstruction_fn(x, name):
  image_x = tf.reshape(x * 255, [1, 1, -1, 1])
  return tf.summary.image(name, tf.cast(image_x, tf.uint8), 1)

def state_reconstruction_fn(x, name):
  return tf.summary.image(name, tf.cast(tf.expand_dims(x * 126, 0), tf.uint8), 1)

# model configuration
action_value_kwargs = {
  'hidden_sizes': [64, 32],
  'activation': tf.nn.relu,
  'action_value_reconstruction_fn': action_value_reconstruction_fn,
}
state_embedding_kwargs = {
  'filters': [8, 8],
  'kernel_sizes': [6, 3],
  'strides': [3, 2],
  'activation': tf.nn.relu,
  'latent_hidden_sizes': [128],
  'latent_hidden_activation': tf.nn.relu,
  'rnn_hidden_sizes': [128],
  'rnn_cell_fn': tf.contrib.rnn.GRUCell,
}
stop_criterion_kwargs = {
  'hidden_sizes': [64, 32],
  'activation': tf.nn.relu,
}
next_state_kwargs = {
  'state_reconstruction_fn': state_reconstruction_fn,
}
optimize_kwargs = {
  'learning_rate': 6.25e-4,
  'l2coeff': .5,
  'max_sequence_length': max_sequence_length - 1,
}

doom_cfg = 'rigid_turning.cfg'

resolution = [30, 45]

def create_env_fn(render=False):
  return ResolutionWrapper(
      VizDoomEnv(doom_cfg, repeat_action=2, render=render), resolution=resolution)

dummy_env = create_env_fn()
state_shape = resolution + [1,]
num_actions = len(dummy_env.unwrapped._actions)
dummy_env.close()

state_space = ContinuousSpace(0., 2., state_shape)
action_space = CategoricalSpace([num_actions], is_one_hot=False)

optimize_state = False # to train, change either/both of these to True
optimize_action_value = False

model = VisualImitationModel(
    state_space, action_space,
    optimize_state=optimize_state,
    optimize_action_value=optimize_action_value).build(
        action_value_kwargs=action_value_kwargs,
        stop_criterion_kwargs=stop_criterion_kwargs,
        state_embedding_kwargs=state_embedding_kwargs,
        goal_state_embedding_kwargs=state_embedding_kwargs,
        next_state_kwargs=next_state_kwargs,
        optimize_kwargs=optimize_kwargs)


total_train_steps = 200000
total_eval_steps = 512
max_goal_distance = 8
min_goal_steps = 10
stop_criterion_threshold = .10

sample_action_idx_op = action_space.build_sample_op(tf.random_uniform([num_actions], -1., 1.))
sample_action_value_op = tf.one_hot(sample_action_idx_op, num_actions)

def step_fn(states, internals):
  sampled_action_idx, sample_action_value = sess.run(
      (sample_action_idx_op, sample_action_value_op))
  return sampled_action_idx, sample_action_value, None


save_every_steps = 1000
ckpt = 'ckpt/vizdoom'
target_ckpt = ckpt
# target_ckpt = 'new_ckpt/vizdoom'


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()

  try:
    latest_ckpt = tf.train.latest_checkpoint(os.path.abspath(os.path.join(ckpt, os.pardir)))
    saver.restore(sess, latest_ckpt)
  except Exception as e:
    print(e)
    pass
  else:
    print('loaded {}'.format(latest_ckpt))

  if optimize_state or optimize_action_value:
    summary_writer = tf.summary.FileWriter('logs')

    capacity = batch_size * 120
    ram = FIFOMemory(
        tf.float32, state_shape,
        tf.int32, [],
        tf.float32, [num_actions],
        capacity)

    dataset = rollout_dataset(ram, batch_size=batch_size, max_sequence_length=max_sequence_length)

    def record_loop():
      pool = RolloutPool(
          create_env_fn,
          num_envs=num_envs,
          num_threads=2)
      while True:
        pool(
            ram,
            step_fn=step_fn,
            synchronous=True,
            num_episodes=num_episodes,
            max_rollout_steps=max_rollout_steps,
            max_trajectory_steps=max_trajectory_steps)
        sleep(.5)

    record_thread = threading.Thread(target=record_loop)
    record_thread.daemon = True
    record_thread.start()

    zero_states = model.zero_state_op(batch_size)

    for steps_taken in trange(total_train_steps):
      (state, next_state, _, action_value, _, _, sequence_length) = sess.run(dataset)

      _, vals, summary = sess.run(
          (model.optimize_op, model.action_value_op, model.summary_op),
          feed_dict=model.optimize_feed_dict(
              state, next_state, next_state,
              action_value, sequence_length,
              initial_state=sess.run(zero_states)))

      global_step_value = tf.train.global_step(sess, model.global_step)
      if (steps_taken + 1) % save_every_steps == 0:
        saver.save(sess, target_ckpt, global_step=global_step_value)

      summary_writer.add_summary(
          summary, global_step_value + steps_taken)
  else:
    zero_states = model.zero_state_op(1)

    env = create_env_fn(render=True)
    _, demos = DemoRecorder(keymap_path='doom_keymap.json').play(
        env, num_actions, max_sequence_length=10, max_episodes=1)
    env.close()

    goal_states = []
    for demo in demos:
      next_goal_states, _, _, _, _, _  = zip(*demo.transitions)
      goal_states.append(next_goal_states[-1])

    env = create_env_fn()
    next_state = env.reset()
    internal_states = None

    states = [next_state]
    state_preds = [np.squeeze(np.zeros_like(next_state))]
    stop_criterions = [0.]

    initial_state = sess.run(zero_states)

    current_goal_state = goal_states.pop(0)
    all_goal_states = [current_goal_state]

    for steps_taken in trange(total_eval_steps):
      state = next_state
      payload = sess.run(
          (model.action_op,
           model.action_value_op,
           model.stop_criterion_op,
           *model.internal_state),
          feed_dict=model.action_value_feed_dict(
              state, current_goal_state,
              initial_state=initial_state))
      action, action_values, stop_criterion = payload[:3]
      next_initial_state = payload[3:]

      action = np.squeeze(action)
      action_values = np.squeeze(action_values)
      stop_criterion = np.squeeze(stop_criterion)
      stop_criterions.append(stop_criterion)

      next_state_pred = sess.run(
          (model.next_state_op),
          feed_dict=model.next_state_feed_dict(
              state, action_values,
              initial_state=initial_state))

      next_state_pred = np.squeeze(next_state_pred)
      state_preds.append(next_state_pred)

      next_state, _, terminal, _ = env.step(action)
      states.append(next_state)
      initial_state = next_initial_state
      if stop_criterion <= stop_criterion_threshold:
        if len(goal_states) == 0:
          break
        current_goal_state = goal_states.pop(0)

      all_goal_states.append(current_goal_state)
      if terminal:
        break

    env.close()

    f = plt.figure()
    f.add_subplot(1, 3, 1)
    f.add_subplot(1, 3, 2)
    f.add_subplot(1, 3, 3)

    def plot_fn(i):
      f.clf()
      ax1 = f.add_subplot(1, 3, 1)
      ax2 = f.add_subplot(1, 3, 2)
      ax3 = f.add_subplot(1, 3, 3)
      ax1.set_title('state [stop = {:.2f}]'.format(stop_criterions[i])); ax1.set_yticks([])
      ax2.set_title('state preds'); ax3.set_yticks([])
      ax3.set_title('goal states'); ax2.set_yticks([])
      ax1.imshow(np.squeeze((states[i] * 126).astype(np.uint8), -1), cmap='gray')
      ax2.imshow((state_preds[i] * 126).astype(np.uint8), cmap='gray')
      ax3.imshow(np.squeeze((all_goal_states[i] * 126).astype(np.uint8), -1), cmap='gray')

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=-1)
    ani = animation.FuncAnimation(f, plot_fn,
                                  frames=np.arange(len(states)),
                                  interval=5)
    # plt.show() # toggle this for just viewing, it's much faster than saving
    ani.save('test_doom.mp4', writer=writer)
