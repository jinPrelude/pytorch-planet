# Tips and Tricks

여기서는 **aha!**, **gotcha!**, 그리고 **touché!** 순간들을 정리해보려고 한다. 😄

- **General**
   1. `configs.yaml` 파일을 사용하면 `argparse`를 수동으로 설정하는 것보다 훨씬 편리하다.  
      서로 다른 파라미터 조합(train, debug, test, gymnasium, dm-control 등)에 대한 기본값을 저장할 수 있다.
   2. `gymnasium`이나 `dm-control` 같은 환경의 기본 동작을 변경할 때는 직접 수정하는 대신 **wrapper**를 사용하자.
   3. `namedtuple`을 활용하면 여러 개의 속성을 깔끔하게 관리할 수 있다.

- **Technical**
   1. `dm-control`의 reward는 0.0과 1.0 사이로 제한되어 있다.  
      그런데 평균이 0이고 분산이 1인 Gaussian 분포로 모델링하는 게 이상해 보일 수 있다.  
      하지만 중요한 점은 **training 시에는 그 분산을 사용하지만, test 시에는 predicted mean을 그대로 사용**한다는 것이다.
   2. Training 중에는 샘플링된 값(예: pixel 값, reward 등)을 clip 하지 말자.  
      하지만 **planning 및 test 시에는 반드시 valid range로 clip**해야 한다.
   3. `torch.distributions.Independent`를 사용하면 다변량 Gaussian 분포를 다루기 편리하다.  
      [이 블로그](https://bochang.me/blog/posts/pytorch-distributions/)와  
      [PyTorch 공식 문서](https://pytorch.org/docs/stable/distributions.html#independent)를 참고해보자.
   4. Loss를 계산할 때는 **batch와 temporal dimension을 기준으로 average를 구해야 한다**.  
      ([ref](https://github.com/google-research/planet/issues/28#issuecomment-487433102))  
      특히 reconstruction loss를 계산할 때는, 모든 픽셀과 채널에 대해 squared error를 합산한 후  
      batch와 time dimension을 기준으로 평균을 내는 것이 일반적이다.
   5. KL-divergence는 **posterior → prior 방향**으로 계산해야 한다.  
      즉, prior을 posterior로 학습시키면서 동시에 posterior를 prior로 regularization한다는 개념이다.  
      이후에는 [KL-balancing](https://arxiv.org/pdf/2010.02193.pdf) 같은 기법을 사용해서  
      prior 쪽의 KL-loss를 더 빠르게 줄이도록 조정할 수도 있다.
   6. **Free nats는 평균 KL-loss가 아니라, 각 prior-posterior 쌍별로 적용해야 한다**.  
      ([ref](https://github.com/google-research/planet/issues/28#issuecomment-487373263))
   7. `rsample()`을 사용하면 gradient가 샘플링 과정까지 전달될 수 있다.  
      반면, `sample()`은 `log_prob()`와 함께 REINFORCE를 구현할 때 사용된다.  
      더 자세한 내용은 [이 문서](https://pytorch.org/docs/stable/distributions.html#score-function)를 참고하자.
   8. CNN 기반 encoder에서는 **`MaxPooling2d`는 공간 불변성을 유지하는 데 사용되지만,  
      `stride`를 활용하면 더 aggressive하게 dimension을 줄일 수 있다**.  
      여기서는 `stride`를 활용하는 방식이 더 적절하다.
   9. TransposedCNN 기반 decoder에서는 `stride`를 활용하는 것이 `Upsample()`보다 더 유연하다.  
      `stride`는 expansion을 위한 적절한 weight를 학습하는 반면,  
      `Upsample()`은 미리 정의된 함수를 그대로 적용하기 때문이다.
   10. 신경망이 stochastic state의 표준 편차를 예측할 때,  
       `softplus(pre_std + 0.55)`를 사용하면 값이 1에 가까운 variance를 유지할 수 있다.
   11. RNN을 사용할 경우, gradient explosion을 방지하기 위해 `grad_norm_clip()`을 적용하자.
   12. 모델에 observation을 입력할 때, zero-mean Gaussian noise(작은 white noise)를 추가하면  
       학습 과정에서 더 일반화된 성능을 기대할 수 있다.