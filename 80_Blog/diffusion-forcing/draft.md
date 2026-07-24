---
title: "To condition my video model on a frame, I set its noise level to zero"
slug: diffusion-forcing
type: blog-draft
status: drafting
created: 2026-06-24
last_updated: 2026-06-24
deliverable: exploratory
sources:
  - "[[../../30_Knowledge/theory/diffusion-forcing]]"
  - "[[../../30_Knowledge/theory/diffusion-forcing.html]]"
  - "[[../../20_Tickets/feat-wan22-ti2v-diffusion-forcing-i2v]]"
  - "paper: Chen et al., 'Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion', NeurIPS 2024"
published_url:
---

# To condition my video model on a frame, I set its noise level to zero

My evaluation videos looked like fog.

I was building an action-conditioned world model on top of a frozen text-to-video
diffusion model. The plan was reasonable: freeze the big pretrained video model,
train a small adapter that takes the agent's action, and let the pair predict
what happens next. The losses went down. And then I looked at the actual samples
and got a uniform grey soup with a faint suggestion of a robot arm somewhere in
the murk.

It took me an embarrassingly long time to admit *why*. The way I'd set it up, the
model had to generate the **entire** scene — table, arm, object, lighting — out of
pure noise, with nothing to go on but a four-number action vector. That is not a
world model. That is a text-to-video model being asked to invent a world from a
whisper. The fog was the honest answer to an impossible question.

What I actually wanted was obvious in hindsight: the model should *start from the
frame the agent is looking at right now* and only predict the consequences of the
action. Condition on the current observation. Predict the future.

So how do you feed a frozen video model a starting frame?

## The textbook answers were dead ends

I knew the standard tricks for image-to-video. There are basically two.

You can add an **image encoder** — run the conditioning frame through CLIP and
feed those features into the model's cross-attention, the way a text prompt goes
in. Or you can **concatenate** the encoded frame, as extra channels, onto the
noisy latent the model denoises, so every layer sees "here's the frame you're
continuing from."

Both work. Both are also off the table when your base is *frozen*. They each
change the model's inputs — new cross-attention, or a wider first convolution —
and a model pretrained without those inputs simply has no weights for them. The
image-native sibling of my model *did* have all that machinery built in, but it
was roughly an order of magnitude bigger, the kind of thing you do not casually
fine-tune on a single GPU.

I was stuck on a frame that I couldn't get into the model. Then I read about
diffusion forcing, and the whole framing flipped.

## The thing I'd been holding fixed

Here is the fact I had never questioned. In every diffusion model I'd used, there
is one noise level — one timestep — for the whole sample. You corrupt everything
by the same amount, and you denoise everything together. For a video, that means
all the frames share a single number that says how noisy the clip currently is.

Diffusion forcing (Chen et al., NeurIPS 2024) asks a small, almost cheeky
question: what if each frame had its *own* noise level?

```
ordinary diffusion:   σ = [0.9, 0.9, 0.9, 0.9, 0.9]   one number for the clip
diffusion forcing:    σ = [0.0, 0.6, 0.75, 0.85, 0.9] one number per frame
                            ^this frame is clean       ^these are still noisy
```

Train the model so it can denoise each frame given its own noise level and the
other frames at *whatever* noise level they happen to be sitting at, and something
quietly profound happens. The noise level stops being a nuisance parameter and
becomes a **dial for how much of each frame is given to you**.

A frame at noise level zero is not noisy at all. It's clean. It's *known*. You're
not asking the model to generate it — you're handing it over. A frame at noise
level one is pure noise: generate it from scratch. And everything in between is a
partial hint.

So my impossible image-to-video problem has an almost insulting answer. Take the
observation frame, drop its noise level to **0**, leave the rest of the clip
noisy, and run the normal denoising loop. The clean frame anchors the generation —
the other frames, through ordinary attention, denoise *toward* something
consistent with it. No image encoder. No extra channels. Nothing added to the
frozen model. The conditioning is a number on the noise schedule.

## Teacher forcing, but with a dimmer switch

If this rings a bell, it should. It's a continuous version of **teacher forcing** —
the old trick from sequence models where, during training, you feed the
ground-truth past and ask for the next token. Teacher forcing is a light switch:
a token is either given or it isn't. Diffusion forcing puts a dimmer on it. The
noise level *is* "how forced" a frame is: zero is fully given, one is fully
generated, and the in-between is a noisy hint you half-trust.

And once "given vs generate" is just a schedule of numbers, the tasks I thought of
as different models collapse into one set of weights:

| What you want | The schedule you pick at sampling time |
|---|---|
| Text → video | every frame the same noise level, annealing to zero together |
| **Image → video** | first frame pinned at 0; the rest anneal from noise |
| Autoregressive / endless video | past frames at 0, future noisier the further out; slide the window |

Same network. You're not switching architectures — you're choosing a noise
trajectory. That's why the model I ended up using is a *unified* text-and-image
model: text-to-video and image-to-video were never two models. They were two
schedules.

## What it actually costs (because nothing is free)

I want to be honest, because "conditioning for free" is the kind of phrase that
gets you in trouble.

Switching to this changes the **training objective**, not just inference. You now
sample noise levels per frame, and — crucially — you only compute the loss on the
frames you're actually predicting. A frame you've handed over at noise level zero
gets *no* gradient; supervising it is meaningless, since its "target" refers to a
noise draw the model never sees. Get that masking wrong and training quietly rots.

At sampling time there's a small discipline too: the denoiser nudges *every*
frame each step, so you have to keep re-pinning the observation frame back to the
clean latent, or it slowly drifts off the very thing it's supposed to be anchored
to.

And there's a gap between the elegant story and what you usually run. The full
idea is *every frame independent*. In practice, for plain image-to-video, I use a
boring **two-level** schedule: the observation at zero, and all the future frames
sharing one noise level that anneals down together. The fully-independent version
earns its keep for long, autoregressive rollouts — where you want frames near
"now" to be cleaner than frames far in the future, so uncertainty grows with the
horizon — which is a different experiment than the one I needed today.

## The lesson I'm taking

The fog cleared the moment I stopped asking the model to invent a world and
started handing it the present. But the part that stuck with me wasn't the result;
it was the *move*. I'd been reaching for a new module — an encoder, a channel,
more parameters — to add a capability. The capability was already sitting inside
the training objective. I just had to take a knob I'd always held fixed, the noise
level, and let it vary per frame.

I think there's a general shape there, and I'll be looking for it now. When you
want a model to do something new, before you bolt a module onto it, check whether
the thing you want is some axis of the existing objective you've quietly been
pinning to a constant. Sometimes the new behaviour isn't a new network. It's a
number you forgot you were allowed to change.

---

*There's an [interactive version of the schedule diagrams](../../30_Knowledge/theory/diffusion-forcing.html)
— drag a frame's noise level and watch it dissolve into static, or hit play to
watch image-to-video denoise with the first frame pinned. The
[theory note](../../30_Knowledge/theory/diffusion-forcing) has the math and the
exact code path.*
