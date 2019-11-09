clear all;
run('vlfeat-0.9.21/toolbox/vl_setup');
categories = dir('256_ObjectCategories');
categories={categories.name};
% categories0={categories{4},categories{7},categories{24}};
categories0=categories(3:length(categories));
categories=categories0;
imgSets=[];
for i =1:length(categories)
    categories{i}=categories{i}(5:length(categories{i}));
    imgSets=[imgSets,imageSet(fullfile('256_ObjectCategories',categories0{i}))];
end
% imgSets=partition(imgSets,80);
[train,test]=partition(imgSets,0.3);
trainnum=0;
testnum=0;
for i=1:length(categories)
    trainnum=trainnum+train(i).Count;
    testnum=testnum+test(i).Count;
end
trainpath=cell(trainnum,1);
testpath=cell(testnum,1);
trainlabel=cell(trainnum,1);
testlabel=cell(testnum,1);
traincount=1;
testcount=1;
for i=1:length(categories)
    for j=1:train(i).Count
        trainpath{traincount}=train(i).ImageLocation{j};
        trainlabel{traincount}=categories{i};
        traincount=traincount+1;
    end
    
    for j=1:test(i).Count
        testpath{testcount}=test(i).ImageLocation{j};
        testlabel{testcount}=categories{i};
        testcount=testcount+1;
    end
end

k=512;
features=[];
featurenum=zeros(1,trainnum);
for i=1:trainnum
    image=imread(trainpath{i});
    if max(size(image,1),size(image,2))>512
        if size(image,1)>size(image,2)
            image=imresize(image,[512 NaN]);
        else
            image=imresize(image,[NaN 512]);
        end
    end
    
    if size(image,3)>1
        image=rgb2gray(image);
    end
    [f,d]=vl_dsift(single(image),'step',8,'size',16);
    features=[features,d];
    featurenum(i)=size(f,2);
end

[dic,index]=vl_kmeans(double(features),k);

h=zeros(trainnum,k);
count=1;
for i=1:trainnum
    for j=1:featurenum(i)
        h(i,index(count))=h(i,index(count))+1;
        count=count+1;
    end
end


forest = vl_kdtreebuild(dic);
testhist=zeros(testnum,k);
for i=1:testnum
    image=imread(testpath{i});
    if max(size(image,1),size(image,2))>512
        if size(image,1)>size(image,2)
            image=imresize(image,[512 NaN]);
        else
            image=imresize(image,[NaN 512]);
        end
    end
    if size(image,3)>1
        image=rgb2gray(image);
    end
    [f,d]=vl_dsift(single(image),'step',8,'size',16);
    [testindex,testdist]=vl_kdtreequery(forest,dic,double(d));
    for j=1:length(testindex)
        testhist(i,testindex(j))=testhist(i,testindex(j))+1;
    end
end


Y=zeros(trainnum,1);
for i=1:length(categories)
    for j=1:trainnum
        if strcmp(trainlabel(j),categories(i))
            Y(j)=i;
        end
    end
end

testY=zeros(testnum,1);
for i=1:length(categories)
    for j=1:testnum
        if strcmp(testlabel(j),categories(i))
            testY(j)=i;
        end
    end
end
addpath('/Applications/MATLAB_R2018a.app/toolbox/libsvm-3.23/matlab');

model=libsvmtrain(Y,h);

[predict,accuracy,prob]=libsvmpredict(Y,h,model);
[predict,accuracy,prob]=libsvmpredict(testY,testhist,model);