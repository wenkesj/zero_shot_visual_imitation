# zero_shot_visual_imitation

Effecient zero-shot goal planning.

```sh
# dependencies
sudo pip install gym matplotlib numpy scipy tensorflow tqdm vizdoom git+https://github.com/wenkesj/alchemy
```

<div align="center">
  <img src="/test_doom.gif"/>
  <p align="center">
    <strong>Demo</strong>
  </p>
</div>

## Methods

The zero-shot "visual" imitation model is broken into 3 policy components that work together to model
the environment, without the use of a supervised signal, such as reward. The 3 policy components are:
**Forward Consistency (FC)**, **Skill Function (SF)**, and **Stop Criterion (SC)**. All these policy
components are parameterized by a shared DNN "body", which acts as a _contextual_ embedding of
the environment in time, and 3 unique DNN "heads" that correspond to each of the components and their
rules.

### The Body

The body acts as a _contextual_ embedding operator on the environment states, `ɸ(s[t])`, learning
abstract features and patterns represented by the state `s[t]`. This is used by all the components.

### Forward Consistency (FC)

The **FC** policy, `s'[t+1] = f(s[t], a[t])`, predicts the next state, `s[t+1]`, given the current
state, `s[t]`, and action `a[t]` that was taken to generate `s[t+1]`. The loss is the distance
`||s'[t+1], s[t+1]||`.

### Skill Function (SF)

The **SF** policy, `a'[t] = g(s[t], s[t+1])`, predicts the action, `a[t]`, that was taken given
the current state, `s[t]`, and the next state, `s[t+1]`. The loss is the cross entropy
`-Σ a[t]*log(a'[t])`.

### Stop Criterion (SC)

The **SC** policy, `P(s*|s[t]) = h(ɸ(s*), ɸ(s[t]))`, plays a crucial role in goal evaluation of
the model by predicting the confidence of when `ɸ(s[t])` _contextually_ matches the goal state embedding,
`ɸ(s*)`, where `s*` could be any number of timesteps away, `t* >= t`. In other words, the **SC**
measures how likely the agent should stop searching for `s*` given `s[t]` at time `t`. Since the
model is never trained, specifically, on a desired goal `s*` that is `t* - t` steps away, the loss
is the **l2 norm** of `s*` and `s[t]`, `||s*, s[t]|| / max||s*, s[t]||`.

## Results

Currently, the only exploits I've done are on **VizDoom**. I've trained the autoencoder model for
~80K steps on the **VizDoom** `my_way_home.wad` scenario and jointly trained the skill model for
~20K steps. The stop criterion model is trained to produce a continuous value from **[0, 1]**
corresponding to how confident the model is that the current state is spatially near the goal state.

To evaluate the model, another completely different scenario is chosen and the agent still performs
suprisingly really well.

The model tends to get stuck when the goal is far away, but this could be due to other factors
(i.e. limited training samples ~100K-20K splits for state/action_value heads), limited resolution
of the environment observation state.

To test and run the simple experiment, run:

```sh
git clone https://github.com/wenkesj/zero_shot_visual_imitation.git
cd zero_shot_visual_imitation
sudo python test_doom.py
```

You need to run this under `sudo` since it uses the pynput tracker to record the demonstration.

## Conclusion

Hurridly, I wrote this experiment over a weekend to accomplish similar results to the ICLR 2018:
["Zero-Shot Visual Imitation"](https://openreview.net/pdf?id=BkisuzWRW), and am very happy with the
simplicity and generalization capability. I extended the original work to a completely
"off-policy" actor-critic model, but the exploration is still a little funky. Inherently, the
simplest choice for the actor is a random exploration model, but this could be changed to a more
intuitive model, such as another learned agent. Maybe I will touch on this later.

There are a lot of kinks to this, but I hope to be able to finish extending the model to not only
visual contexts, applying to continuous action domains, modify the supporting
[library](https://github.com/wenkesj/alchemy) to fit all gym observation/action spaces, such as
possible compatibility with other rl libraries, such as [anyrl](https://github.com/unixpickle/anyrl-py).

## License

The test VizDoom .wad/.cfg comes from "DeepDoom: Navigating Complex Environments Using Distilled Hierarchical Deep Q-Networks"

```
The MIT License (MIT)

Copyright (c) 2018 Sam Wenke

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
