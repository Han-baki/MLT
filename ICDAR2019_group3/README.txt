--------------------------------------------------------------------------------------------------------------

3조의 작업을 tbpp_train.ipynb 파일에 담았습니다. Google colab의 GPU 환경에서 학습을 실행하였습니다.
각 Cell들의 맨 위에, page0부터 page5로 주석을 달았습니다. 

ICDAR2019 MLT data로 학습을 진행한 후, ICDAR2017 validation data로 모델 평가를 진행했습니다.

ICDAR2019 MLT data 다운로드 : https://rrc.cvc.uab.es/?ch=15&com=downloads 에 들어가서 로그인 후, Task1에 있는 data를 다운로드
ICDAR2017 validation data 다운로드 : https://rrc.cvc.uab.es/?ch=8&com=downloads 에 들어가서 로그인 후, Task1의 Validation Set을 다운로드
두 data를 다운로드 후, 압축을 풀고 data들의 directory path들을 page0에 입력합니다.

page0에 있는 것이 input입니다. input에 다음과 같이 집어넣어주고 run all을 누르면 파일이 전부 돌아가게 했습니다.
train_ImagesPart1_dir : ICDAR2019 ImagesPart1의 image파일이 있는 directory
train_ImagesPart2_dir : ICDAR2019 ImagesPart2의 image파일이 있는 directory
train_gt_dir : ICDAR2019 train_gt_t13의 text파일이 있는 directory
test_img_dir : ICDAR2017 validation data의 image파일이 있는 directory
test_gt_dir : ICDAR2017 validation data의 text파일이 있는 directory

--------------------------------------------------------------------------------------------------------------

page1부터 page5까지의 작업을 순서대로 나열하면,

1. train, test data 만들기 (page1, page2)
2. model training (page3)
3. threshold값의 grid search (page4)
4. 최종 결과 (page5)

이렇게 네 단계로 나뉩니다.

--------------------------------------------------------------------------------------------------------------

page1에서는 
2019 data 10000개에서 54개의 (에러)이미지를 제외한 후 9005개, 941개로 나누고,
9005개를 train(directory)에, 941개를 test(directory)에 각각 저장합니다. 

page2에서는
train 9005개 =>ICDAR2019_train.pkl
test 941개 => ICDAR2019_test.pkl
2017 데이터 1800개 => ICDAR2017_validation.pkl
이렇게 pkl 파일로 저장합니다.

page3에서는
ICDAR2019_train.pkl 파일을 불러와서 40 epoch train을 수행합니다.
1~20 epoch은 learning rate = 5*e-4
21~40 epoch은 learning rate = 2*e-4 로 수행합니다.
첫 에폭이 오래걸리고, 두번째 에폭부터는 1에폭당 22분정도 걸립니다.

page4와 page5에서는 f1 score가 계산되는 부분입니다.
여기에 weights_path라는 기본 None으로 설정된 변수가 있는데, 이것을
모델에 load하고싶은 weight의 path로 설정해주면 load한 weight로 f1 score를 계산하게 됩니다.
weights_path를 None값으로 그대로 놓으면, page3에서 training했던 모델을 그대로 가져와서 씁니다.
checkpoints 폴더에 저희가 train 시켜서 저장했던 weights.040.h5파일을 올려놓았으니,
weights_path = 'checkpoints/weights.040.h5'로 설정하면 
저희가 학습시킨 weight로 계산된 f1 score로 볼 수 있습니다.

page4에서는
ICDAR2019_test.pkl을 불러와서 random으로 500개의 data를 가져온 후,
confidence threshold와 f1score에 대한 그래프를 그립니다.

page5에서는
ICDAR2017_validation.pkl을 불러와서 1800개의 data를 전부 가져온 후,
text detection에 대한 precision, recall, f1score과
text detection + classification에 대한 precision, recall, f1score를 출력합니다.

--------------------------------------------------------------------------------------------------------------
