clear all;
run('vlfeat-0.9.21/toolbox/vl_setup');
categories = dir('256_ObjectCategories');
categories={categories.name};
% categories0={categories{4},categories{7}};
categories0={categories{4},categories{7},categories{24}};
% categories0=categories(3:length(categories));
categories=categories0;
imgSets=[];
for i =1:length(categories)
    categories{i}=categories{i}(5:length(categories{i}));
    imgSets=[imgSets,imageSet(fullfile('256_ObjectCategories',categories0{i}))];
end
imgSets=partition(imgSets,80);
[train,test]=partition(imgSets,60);
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


features=[];
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
end

k=2048;
[dic,index]=vl_kmeans(double(features),k);
forest = vl_kdtreebuild(dic);


h=zeros(trainnum,k);
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
    height=size(image,1);
    width=size(image,2);
    for j=0:2
        temp_width=floor(width/(2^j));
        temp_height=floor(height/(2^j));
        for m=1:2^j
            for l=1:2^j
                imgtemp=image((l-1)*temp_height+1:l*temp_height,(m-1)*temp_width+1:m*temp_width);
                [f,d]=vl_dsift(single(imgtemp),'step',8,'size',16);
                trainindex=vl_kdtreequery(forest,dic,double(d));
                for n=1:length(trainindex)
                    h(i, trainindex(n))=h(i,trainindex(n))+1/2^j;
                end
            end
        end
    end
end

% for i=1:size(h,1):
%     h(i,:)=h(i,:)./sum(h(i.:));
% end

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
    height=size(image,1);
    width=size(image,2);
    for j=0:2
        temp_width=floor(width/(2^j));
        temp_height=floor(height/(2^j));
        for m=1:2^j
            for l=1:2^j
                imgtemp=image((l-1)*temp_height+1:l*temp_height,(m-1)*temp_width+1:m*temp_width);
                [f,d]=vl_dsift(single(imgtemp),'step',8,'size',16);
                testindex=vl_kdtreequery(forest,dic,double(d));
                for n=1:length(testindex)
                    testhist(i, testindex(n))= testhist(i, testindex(n))+1/2^j;
                end
            end
        end
    end
end

% for i=1:size(testhist,1):
%     testhist(i,:)=testhist(i,:)./sum(testhist(i.:));
% end


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