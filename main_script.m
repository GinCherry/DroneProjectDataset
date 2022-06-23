% Download Pretrained Detector

Tfile = load('YoloTarget7.mat','detector');
detector = Tfile.detector;

scores = detector.Fitted.Probability;
[X,Y,T,AUC] = perfcurve(

%{ 

Load the Dataset

Dfile = load('dataset_gTruth7.mat');
data = Dfile.dataset;
targetDataset = data;  


% Add the fullpath to the local vehicle data folder.

targetDataset.Filename = fullfile(pwd,targetDataset.Filename);

 
Split the dataset into training, validation, and test sets. 
Select 60% of the data for training, 10% for validation, 
and the rest for testing the trained detector. 


rng(0);
shuffledIndices = randperm(height(targetDataset));
idx = floor(0.6 * length(shuffledIndices) );

trainingIdx = 1:idx;
trainingDataTbl = targetDataset(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl = targetDataset(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = targetDataset(shuffledIndices(testIdx),:);

%{
Use imageDatastore and boxLabelDatastore to create datastores 
for loading the image and label data during training and evaluation.
%}

imdsTrain = imageDatastore(trainingDataTbl{:,'Filename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'Target'));

imdsValidation = imageDatastore(validationDataTbl{:,'Filename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,'Target'));

imdsTest = imageDatastore(testDataTbl{:,'Filename'});
bldsTest = boxLabelDatastore(testDataTbl(:,'Target'));

% Combine image and box label datastores.

trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);
%}
%{ 
Display one of the training images and box labels.

data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

% Create a YOLO v2 Object Detection Network


First, specify the network input size and the number of classes. 
When choosing the network input size, consider the minimum size 
required by the network itself, the size of the training images, 
and the computational cost incurred by processing data at the selected size. 
When feasible, choose a network input size that is close to the size 
of the training image and larger than the input size required for the network. 
To reduce the computational cost of running the example, 
specify a network input size of [224 224 3], 
which is the minimum size required to run the network.
%}

inputSize = [224 224 3]; 

%{
 Define the number of object classes to detect. (1 class = target)

numClasses = width(targetDataset)-1;

%{ 
Note that the training images are bigger than 224-by-224 
and vary in size, so we must resize the images 
in a preprocessing step prior to training.
%}

% Pre-process Data

trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 7;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);

% ESTIMATE ANCHOR BOXES:
% https://it.mathworks.com/help/vision/ug/estimate-anchor-boxes-from-training-data.html

% use resnet50 to load a pretrained ResNet-50 model

featureExtractionNetwork = resnet50;
featureLayer = 'activation_40_relu'; % EMPIRICALLY CHOSEN: can be changed

% Create the YOLO v2 Object Detection Network

lgraph = yolov2Layers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

% DATA AUGMENTATION

augmentedTrainingData = transform(trainingData,@augmentData);

% Visualize the augmented images.
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
    reset(augmentedTrainingData);
end
% figure
% montage(augmentedData,'BorderSize',10)

% Preprocess Augmented Training Data

preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
preprocessedValidationData = transform(validationData,@(data)preprocessData(data,inputSize));

data = read(preprocessedTrainingData);

% Display the image and bounding boxes
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
% figure
% imshow(annotatedImage)


%{

TRAINING 

options = trainingOptions('sgdm', ...
        'MiniBatchSize',16, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',20, ... 
        'CheckpointPath',tempdir, ...
        'ValidationData',preprocessedValidationData);
    
% since set the doTraining to 'true'    
[detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);

% save the trained model

save('YoloTarget7.mat','detector')


%}
%}

% BENCHMARK 0 and computation of the centre

I_0 = imread('benchmarks\bench_0.jpg');
I_0 = imresize(I_0,inputSize(1:2));
[bboxes,scores] = detect(detector,I_0);                    

I_0 = insertObjectAnnotation(I_0,'rectangle',bboxes,scores);

% Coordinates of the target centre
x_coord = bboxes(1) + (bboxes(3)/2);
y_coord = bboxes(2) + (bboxes(4)/2);

centre = [x_coord y_coord]

figure
imshow(I_0)
axis on
hold on;
title('Computation of the centre');

% Plot cross in the centre
plot(x_coord, y_coord, 'r+', 'MarkerSize', 10, 'LineWidth', 2);

% BENCHMARK 1

I_1 = imread('benchmarks\bench_1.jpg');
I_1 = imresize(I_1,inputSize(1:2));
[bboxes1,scores1] = detect(detector,I_1);                    

I_1 = insertObjectAnnotation(I_1,'rectangle',bboxes1,scores1);

% BENCHMARK 2

I_2 = imread('benchmarks\bench_2.jpg');
I_2 = imresize(I_2,inputSize(1:2));
[bboxes2,scores2] = detect(detector,I_2);                    

I_2 = insertObjectAnnotation(I_2,'rectangle',bboxes2,scores2);

% BENCHMARK 3

I_3 = imread('benchmarks\bench_3.jpg');
I_3 = imresize(I_3,inputSize(1:2));
[bboxes3,scores3] = detect(detector,I_3);                    

I_3 = insertObjectAnnotation(I_3,'rectangle',bboxes3,scores3);

% BENCHMARK 4

I_4 = imread('benchmarks\bench_4.jpg');
I_4 = imresize(I_4,inputSize(1:2));
[bboxes4,scores4] = detect(detector,I_4);                    

I_4 = insertObjectAnnotation(I_4,'rectangle',bboxes4,scores4);

% BENCHMARK 5

I_5 = imread('benchmarks\bench_5.jpg');
I_5 = imresize(I_5,inputSize(1:2));
[bboxes5,scores5] = detect(detector,I_5);                    

I_5 = insertObjectAnnotation(I_5,'rectangle',bboxes5,scores5);

% BENCHMARK 6

I_6 = imread('benchmarks\bench_6.jpg');
I_6 = imresize(I_6,inputSize(1:2));
[bboxes6,scores6] = detect(detector,I_6);                    

I_6 = insertObjectAnnotation(I_6,'rectangle',bboxes6,scores6);

% BENCHMARK 7

I_7 = imread('benchmarks\bench_7.jpg');
I_7 = imresize(I_7,inputSize(1:2));
[bboxes7,scores7] = detect(detector,I_7);                    

I_7 = insertObjectAnnotation(I_7,'rectangle',bboxes7,scores7);

% BENCHMARK 8

I_8 = imread('benchmarks\bench_8.jpg');
I_8 = imresize(I_8,inputSize(1:2));
[bboxes8,scores8] = detect(detector,I_8);                    

I_8 = insertObjectAnnotation(I_8,'rectangle',bboxes8,scores8);

% BENCHMARK 9

I_9 = imread('benchmarks\bench_9.jpg');
I_9 = imresize(I_9,inputSize(1:2));
[bboxes9,scores9] = detect(detector,I_9);                    

I_9 = insertObjectAnnotation(I_9,'rectangle',bboxes9,scores9);

% show results
figure
sgtitle('Benchmarks');
subplot(3,3,1), imshow(I_1)
subplot(3,3,2), imshow(I_2)
subplot(3,3,3), imshow(I_3)
subplot(3,3,4), imshow(I_4)
subplot(3,3,5), imshow(I_5)
subplot(3,3,6), imshow(I_6)
subplot(3,3,7), imshow(I_7)
subplot(3,3,8), imshow(I_8)
subplot(3,3,9), imshow(I_9)

% Occlusion Performance 

figure
sgtitle('Occlusions')
subplot(1,2,1), imshow(I_2)
subplot(1,2,2), imshow(I_9)

% Shadows Performance

figure
sgtitle('Shadows')
subplot(1,2,1), imshow(I_7)
subplot(1,2,2), imshow(I_8)

% Deformation Performance

figure
sgtitle('Deformed Shape')
subplot(1,2,1), imshow(I_5)
subplot(1,2,2), imshow(I_3)

% Intense Light Reflection Performance

I_6b = imread('benchmarks\bench_6b.jpg');
I_6b = imresize(I_6b,inputSize(1:2));
[bboxes6b,scores6b] = detect(detector,I_6b);                    

I_6b = insertObjectAnnotation(I_6b,'rectangle',bboxes6b,scores6b);

figure
subplot(1,2,1), imshow(I_6)
subplot(1,2,2), imshow(I_6b)
sgtitle('Intense Light Reflections')

% Darkness Performance

I_4b = imread('benchmarks\bench_4b.jpg');
I_4b = imresize(I_4b,inputSize(1:2));
[bboxes4b,scores4b] = detect(detector,I_4b);                    

I_4b = insertObjectAnnotation(I_4b,'rectangle',bboxes4b,scores4b);

figure
subplot(1,2,1), imshow(I_4)
subplot(1,2,2), imshow(I_4b)
sgtitle('Darkness')

% Distance Performance

I_10 = imread('benchmarks\bench_10.jpg');
I_10 = imresize(I_10,inputSize(1:2));
[bboxes10,scores10] = detect(detector,I_10);                    

I_10 = insertObjectAnnotation(I_10,'rectangle',bboxes10,scores10);

I_11 = imread('benchmarks\bench_11.jpg');
I_11 = imresize(I_11,inputSize(1:2));
[bboxes11,scores11] = detect(detector,I_11);                    

I_11 = insertObjectAnnotation(I_11,'rectangle',bboxes11,scores11);

figure
subplot(1,2,1), imshow(I_11)
subplot(1,2,2), imshow(I_10)
sgtitle('Distance')

%{
preprocessedTestData = transform(testData,@(data)preprocessData(data,inputSize));
detectionResults = detect(detector, preprocessedTestData);
[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, preprocessedTestData);

 
figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f',ap))


% Support Functions

function data = preprocessData(data,targetSize)
% Resize image and bounding boxes to the targetSize.
scale = targetSize(1:2)./size(data{1},[1 2]);
data{1} = imresize(data{1},targetSize(1:2));
boxEstimate=round(data{2});
boxEstimate(:,1)=max(boxEstimate(:,1),1);
boxEstimate(:,2)=max(boxEstimate(:,2),1);
data{2} = bboxresize(boxEstimate,scale);
end

function B = augmentData(A)
% Apply random horizontal flipping, and random X/Y scaling. Boxes that get
% scaled outside the bounds are clipped if the overlap is above 0.25. Also,
% jitter image color.
B = cell(size(A));
I = A{1};
sz = size(I);
if numel(sz)==3 && sz(3) == 3
    I = jitterColorHSV(I,...
        'Contrast',0.2,...
        'Hue',0,...
        'Saturation',0.1,...
        'Brightness',0.2);
end
% Randomly flip and scale image.
tform = randomAffine2d('XReflection',true,'Scale',[1 1.1]);
rout = affineOutputView(sz,tform,'BoundsStyle','CenterOutput');
B{1} = imwarp(I,tform,'OutputView',rout);
% Apply same transform to boxes.
boxEstimate=round(A{2});
boxEstimate(:,1)=max(boxEstimate(:,1),1);
boxEstimate(:,2)=max(boxEstimate(:,2),1);
[B{2},indices] = bboxwarp(boxEstimate,tform,rout,'OverlapThreshold',0.25);
B{3} = A{3}(indices);
% Return original data only when all boxes are removed by warping.
if isempty(indices)
    B = A;
end
end

%%% REF %%%
% https://it.mathworks.com/help/vision/ug/train-an-object-detector-using-you-only-look-once.html

%}

