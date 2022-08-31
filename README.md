**[NEW!] 2022 Ego4D Challenges now open for Hands And Objects**
- [Object State Change Classification](https://eval.ai/web/challenges/challenge-page/1627/overview)
- [PNR Temporal Localization](https://eval.ai/web/challenges/challenge-page/1622/overview)
- [State Change Object Detection](https://eval.ai/web/challenges/challenge-page/1632/overview)



# Ego4D Hands & Objects Benchmark

[EGO4D](https://ego4d-data.org/docs/) is the world's largest egocentric (first person) video ML dataset and benchmark suite.

For more information on Ego4D or to download the dataset, read: [Start Here](https://ego4d-data.org/docs/start-here/).

The [Hands & Objects Benchmark](https://ego4d-data.org/docs/benchmarks/hands-and-objects/) aims to understand the camera-wearers present activity – in terms of interactions with objects.  This repository contains the code needed to reproduce the results in the [Ego4D: Around the World in 3,000 Hours of Egocentric Video](https://arxiv.org/abs/2110.07058).

## Task Definition

The Hands & Objects benchmark captures how the camera-wearer changes the state of an object by using or manipulating it – which we call an object state change. Though cutting a piece of lumber in half can be achieved through many methods (e.g., various tools, force, speeds, grasps, end effectors), all should be recognized as the same state change.  

Object state changes can be viewed along temporal, spatial, and semantic axes, leading to these three tasks:

1. [Point-of-no-return temporal localization](./state-change-localization-classification/README.md): given a short video clip of a state change, the goal is to estimate the keyframe that contains the point-of–no-return (PNR) (the time at which a state change begins)

1. [State change object detection](./state-change-localization-classification/README.md): given three temporal frames (pre, post, PNR), the goal is to regress the bounding box of the object undergoing a state change

1. [Object state change classification](./state-change-localization-classification/README.md): Given a short video clip, the goal is to classify whether an object state change has taken place or not

Please see the individual README for each of the sub-task directories for more detail. 

### License

Ego4D is released under the MIT License.
