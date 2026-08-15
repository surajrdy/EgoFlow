# Experimental video-hand geometry

This lane tests whether public RGB video can add manipulation evidence without
privileged robot state. It is deliberately separate from the frozen EgoFlow model
and blind-test metrics.

The extractor samples at 8 FPS and records normalized 2D palm centers plus a
thumb-to-index aperture proxy. The detector seeks a moving hand that changes
direction sharply before a clear close/open grasp cycle. It rejects stationary
waiting, tracking gaps, and camera-coupled bimanual motion. Every proposal is
tagged `video_hand_geometry_v1` and presented as `ABORTED REACH?`, never as fact.

```bash
uv run modal run hackathon/egoflow/hand_modal.py \
  --episode-ids ID1,ID2 --sample-fps 8
```

The bounded 18-episode probe used eight CPU workers. Hands were present on 94.6%
of sampled frames and the initial strict detector returned 90 review candidates
but missed the human-labeled 8–13 s Angry Bird example. Four global variants were
then compared on the existing 18 caches. Treating aperture as soft evidence and
allowing a brief landmark gap proposes 8.267–10.133 s in the hero span (171.2°
redirection). That choice is validation tuning; its 90 windows have not received
prospective human review.

The missing ingredient is object identity and persistence. A defensible
approach→abort→switch claim needs evidence that the hand approached object A,
object A did not move, and the hand subsequently interacted with object B. A palm
trajectory alone cannot distinguish that from a successful transport or camera
motion. Accordingly, the hero marker is explicitly `HAND / EXPERIMENTAL` and no
frozen metric is changed.
