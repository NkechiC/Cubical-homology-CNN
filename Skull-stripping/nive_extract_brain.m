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
    skullStripped(repmat(~mask, [1 1 3])) = 0;

    % % Save the skull-stripped image
    % imwrite(skullStripped, outputPath);
end
