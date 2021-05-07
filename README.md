# classification_resnet
시작 하기 전, 이 실험은 Tensorfloww Library API을 기반으로 Colab이 제공하는 GPU를 사용하였다. 실험 결과로 (Train, Validation)Accuracy, Loss, feature map, Grad CAM을 확인할 수 있다.

#### - 데이터는 kaggle에서 제공하는 Cats-vs-Dogs 데이터를 이용하여 2만 5천장에서 1만 7천개를 train_data, 4천개를 validation_data, test_data 4000으로 나누어 실험을 진행하였음.

### ResNet(Residual Networks)
ResNet은 2015년 ILSVRC 대회에서 우승한 모델로써 신경망의 구조가 깊어짐에 따라 정확도가 저하되는 문제를 해결하기 위한 방법으로「잔류 학습」을 도입하였다. 간단히 말해, 기존 네트워크를 H(x)(x는 레이어의 입력) F(x) = H(x) - x 라고 하면 네트워크 F(x) + x가 대략 H(x)로 학습되도록 하는 것이다. 즉, 레이어의 입력과 출력 간의 차이를 학습 하고 저하되는 문제를 해결할 수 있었다. 이것은 왼쪽 아래에 있는 그림을 보면 확인할 수 있다. 오른쪽은 그림은 50개 이상 Layer에서 Bottleneck을 이용하는 구조로 건너 뛰는 것을 볼 수 있다. 이는 층이 깊어질수록 연산량이 늘어남에 따라 제안된 방법으로 채널 수가 1 x 1 Conv로 줄어드 다음 3 x 3 Conv를 거쳐 1 x 1 Conv를 다시 통과하여 채널 수를 다시 복구하는 것을 볼 수 있다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/74638207-cdf89080-51ae-11ea-93fc-4f4646158be5.png" width="75%"></p>

다음 아래 그림은 ResNet 논문에서 제안한 5개의 구조다. 다만 실험에서는 101Layer, 50Layer을 사용하였다. 

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/74638228-dbae1600-51ae-11ea-8f6d-ba4e685cf445.png" width="75%"></p>

### 101-layer vs 50-layer
이 데이터를 기준으로 레이어의 깊이에 대해서 비교하여 실험을 진행했으며, 여기서 주로 관점은 레이어층이 깊어질수록 다양한 피쳐 맵을 확인을 하고자 했으며, 그 피쳐 맵이 어디를 바라보는지 확인을 하고싶었다.

### 설정
모델 빌드의 경우 옵티마이저로 SGD(learning rate = 0.001, momentum = 0.9)함수를 사용했으며 손실 함수는 binary_crossentropy로 설정했으며 하이퍼 파라미터는 550epoch, (image_size 224, 224, 3), step(train, validation 90, 20),batch_size(train, validation 20)으로 설정했다.

그리고 효율적으로 모델 성능을 이끌고자 추가적으로 Data Augment을 다음과 같이 사용함.

    - train : rotation_range=40, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, rescale=1./255
    - validation : rescale=1./255
    - test : rescale=1./255

### 학습 결과

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/84604714-f2e9df80-aed2-11ea-9a6f-eeb8855f8737.png" width="65%"></p>

<table border="1">
<th>Train Model Layer</th>
<th>Max Train Accuracy</th>
<th>Min Train Loss</th>
<th>Max Validation Accuracy</th>
<th>Min Val Loss</th>    
<tr>
<td>50_Layer</td>
<td>0.96375</td>
<td>0.0903</td>
<td>0.9475</td>
<td>0.1564</td>    
</tr>    
<tr>
<td>101_Layer</td>
<td>0.9568</td>
<td>0.1212</td> 
<td>0.9375</td>    
<td>0.1708</td>    
</tr>    
</table>    


101 layer 학습 결과 Accuracy 92.70%, Loss 0.1951가 나왔으며, 50 layer 학습 결과 Accuracy 93.80%, Loss 0.1692가 나온 것을 볼 수 있었으며 똑같은 조건으로 두개의 Layer를 비교하였을 때, 이 데이터에는 50 Layer가 데이터에 더 적합 생각했으며 아래 feature map에서 레이어층의 깊이에 보충 설명을 확인할 수 있다.

#### - feature map(conv_2d layer)
Feature map을 원할하게 보기 위한 연산을 적용함으로써 Convolution layer에 해당하는 레이어층의 피쳐 맵을 가져와 시각화 함.

#### 50-layer

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75012532-82532900-54c5-11ea-93c4-2ca3511520c0.png" width="75%"></p>

#### 101-layer

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/74638370-1ca62a80-51af-11ea-9a4c-d18b593d081a.png" width="75%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/74638389-2465cf00-51af-11ea-8884-f7839e414b94.png" width="75%"></p>

레이어층이 깊어질수록 다양하고 세밀한 피쳐 맵을 볼 수 있었으며 바라보는게 다 다르다는 것을 볼 수 있으며 이 피쳐 맵에 대한 Grad CAM을 뒤에서 확인하였다.

#### - Grad CAM:Generalized version of CAM(cat, dog)
간단히 말해서, Grad CAM은 얼굴 위치 추적기라고 부르며 Grad CAM을 원할하게 보기 위한 연산을 적용함으로써  모델이 이미지에 대해서 어느 위치를 보며 예측을 한 것인지 레이어층 별로 알 수 있다.

#### 50-layer

Cat Image Grad CAM

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81383322-d5468f00-914a-11ea-998f-c4e38317af5f.png" width="50%"></p>

Dog Image Grad CAM

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81383363-e394ab00-914a-11ea-8d65-824ff13bf45f.png" width="50%"></p>

위의 이미지를 보면 사람하고 같이 있을 경우 고양이르 제대로 못 보는 경우와 개의 특징을 찾지 못하여 전반적으로 이미지를 봐서 잘 못 예측하는 것을 볼 수 있었다.

#### 101-layer

Cat Image Grad CAM

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81383114-708b3480-914a-11ea-85c0-8d90bb9b37d3.png" width="50%"></p>

Dog Image Grad CAM

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81383184-931d4d80-914a-11ea-8cbb-33cb0bd55f56.png" width="50%"></p>

위 이미지를 보면개와 고양이가 같이 있는 이미지의 경우가 있는데 개의 클래스를 가지고 있지만고양이 다리를 보고 고양이로 보는 경우도 있었다.

#### - 일부 Convolution Layer층을 사용하여 깊이에 따른 Grad CAM의 변화 알아보기
###### 사진이 작아서 안보이면 확대하여 보면 조금 더 자세히 볼 수 있습니다.

#### Layer 50 Grad CAM

올바르게 예측한 경우(cat, dog)

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83741014-a0b6ec00-a692-11ea-913e-81393e24681c.png" width="50%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83740990-95fc5700-a692-11ea-987b-6ac8773440d1.png" width="50%"></p>

잘 못 예측한 경우(cat, dog)

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83741073-b0cecb80-a692-11ea-8737-7dd10ea2ea85.png" width="50%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83741120-bfb57e00-a692-11ea-9f6f-a168803adf19.png" width="50%"></p>

#### Layer 101 Grad CAM

올바르게 예측한 경우(cat, dog)

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83860394-2baee980-a75a-11ea-84e3-ca631ce413b5.png" width="50%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83860428-379aab80-a75a-11ea-84a0-f4cd844707b2.png" width="50%"></p>

잘 못 예측한 경우(cat, dog)

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83860471-41bcaa00-a75a-11ea-929e-1f0e0d7b7e8a.png" width="50%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/83860498-4aad7b80-a75a-11ea-816f-32b86b60f9e3.png" width="50%"></p>

일부 Convolution Layer층을 사용하여 깊이에 따른 Grad CAM의 변화를 확인한 결과로 올바르게 예측한 경우는 cat, dog를 찾아가는 과정이 보이며 잘 못 예측한 경우는 다른 곳을 찾아가서 다르게 예측하거나 다른 사물, 사람, 동물 등 같이 있어서 고유의 클래스를 보는 것이 아닌 다른 곳을 바라보게 되는 경우들이 있었으며 무엇보다도 각 레이어층에서 각자 다른 곳을 바라보고 있는 것을 확인할 수 있으며 레이어층이 더 깊은 모델을 사용하면 더 다양하기도 하며 세밀하게 볼 수 있었지만 레이어층이 깊어 처음에 바라보던 것이 사라지면서 다른 곳을 바라보게 되는 상황이 발생하기도 했다.

