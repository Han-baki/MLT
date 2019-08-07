
3���� �۾��� tbpp_train.ipynb ���Ͽ� ��ҽ��ϴ�.

�ϴ� page0�� �ִ� ���� input�Դϴ�. input�� ������ ���� ����־��ְ� run all�� ������ ������ ���� ���ư��� �߽��ϴ�.
train_ImagesPart1_dir : ICDAR2019 ImagesPart1�� image������ �ִ� directory
train_ImagesPart2_dir : ICDAR2019 ImagesPart2�� image������ �ִ� directory
train_gt_dir : ICDAR2019 train_gt_t13�� text������ �ִ� directory
test_img_dir : ICDAR2017 validation data�� image������ �ִ� directory
test_gt_dir : ICDAR2017 validation data�� text������ �ִ� directory


���� �۾��� ������� �����ϸ�,

1. train, test data ����� (page1, page2)
2. model training (page3)
3. threshold�� grid search (page4)
4. ���� ��� (page5)

�̷��� �� �ܰ�� �����ϴ�.

page1������ 
2019 data 10000������ 54���� (����)�̹����� ������ �� 9005��, 941���� ������,
9005���� train(directory)��, 941���� test(directory)�� ���� �����մϴ�. 

page2������
train 9005�� =>ICDAR2019_train.pkl
test 941�� => ICDAR2019_test.pkl
2017 ������ 1800�� => ICDAR2017_validation.pkl
�̷��� pkl ���Ϸ� �����մϴ�.

page3������
ICDAR2019_train.pkl ������ �ҷ��ͼ� 40 epoch train�� �����մϴ�.
1~20 epoch�� learning rate = 5*e-4
21~40 epoch�� learning rate = 2*e-4 �� �����մϴ�.
ù ������ �� �����ɸ��� �ι�° �������ʹ� 1������ 22������ �ɸ��ϴ�.

page4�� page5������ f1 score�� ���Ǵ� �κ��Դϴ�.
���⿡ weights_path��� �⺻ None���� ������ ������ �ִµ�, �̰���
�𵨿� load�ϰ���� weight�� path�� �������ָ� load�� weight�� f1 score�� ����ϰ� �˴ϴ�.
weights_path�� None������ �״�� ���θ�, page3���� training�ߴ� ���� �״�� �����ͼ� ���ϴ�.
checkpoints ������ ���� train ���Ѽ� �����ߴ� weights.040.h5������ �÷���������,
weights_path = 'checkpoints/weights.040.h5'�� �����ϸ� 
���� �н���Ų weight�� ���� f1 score�� �� �� �ֽ��ϴ�.

page4������
ICDAR2019_test.pkl�� �ҷ��ͼ� random���� 500���� data�� ������ ��,
confidence threshold�� f1score�� ���� �׷����� �׸��ϴ�.

page5������
ICDAR2017_validation.pkl�� �ҷ��ͼ� 1800���� data�� ���� ������ ��,
text detection�� ���� precision, recall, f1score��
text detection + classification�� ���� precision, recall, f1score�� ����մϴ�.