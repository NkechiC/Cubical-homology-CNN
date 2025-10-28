function skullStripped = nive_extract_brain(imagePath, modelPath) %Add outputPath as an argument if you want to save the output
    % Load the pretrained DeepLabV3+ model
    netStruct = load(modelPath);
    net = netStruct.SkullNetMegaDeepLabv3Plus;

    % Read and prepare the image
    img = imread(imagePath);
    img = im2uint8(img);

    % Segment the brain region
    brainMask = semanticseg(img, net);
    mask = brainMask == "brain";

    % Apply the mask to get skull-stripped image
    skullStripped = img;
    
    for c = 1:size(img, 3)
        channel = skullStripped(:,:,c);
        channel(~mask) = 0;
        skullStripped(:,:,c) = channel;
    end

    % % Save the skull-stripped image
    % imwrite(skullStripped, outputPath);
end
