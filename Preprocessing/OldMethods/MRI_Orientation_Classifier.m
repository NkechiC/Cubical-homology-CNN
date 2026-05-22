function output = MRI_Orientation_Classifier(filePath)
    img = imread(filePath);
    [height, width, ~] = size(img);

    img_new = zeros(height, width, 3, 'uint8');

    if size(img,3)==3
        img_new=imresize(img,[224 224]);
    else
        img_new(:,:,1)=img;
        img_new(:,:,2)=img;
        img_new(:,:,3)=img;
        img_new=imresize(img_new,[224 224]);
    end

    load('FINALtwelveclassmodel4epochsMRItypesRotatedMobileNetRGB224.mat');

    output = string(classify(net, img_new));
end