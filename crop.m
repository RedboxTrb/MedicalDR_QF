% Define input and output directories
inputDir = "D:\DREAM Dataset\1030The SUSTech-SYSU dataset for automated exudate detection and diabetic retinopathy grading (1)\originalImages";
outputDir = "D:\DREAM Dataset\1030The SUSTech-SYSU dataset for automated exudate detection and diabetic retinopathy grading (1)\croppedimages";

% Create output folder if it does not exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Define allowed image extensions
imageExts = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff'};

% Collect all image files
imageFiles = [];
for i = 1:length(imageExts)
    imageFiles = [imageFiles; dir(fullfile(inputDir, imageExts{i}))]; %#ok<AGROW>
end

% Process each image
for k = 1:length(imageFiles)
    % Read image
    imgName = imageFiles(k).name;
    imgPath = fullfile(inputDir, imgName);
    img = imread(imgPath);

    % Convert to grayscale
    gray = rgb2gray(img);

    % Thresholding
    binaryImg = gray > 10;
    T = uint8(binaryImg) * 2;

    % Find contours
    [B, L] = bwboundaries(T, 'noholes');

    % Find largest contour
    maxArea = 0;
    maxIndex = 1;
    for i = 1:length(B)
        boundary = B{i};
        area = polyarea(boundary(:,2), boundary(:,1));
        if area > maxArea
            maxArea = area;
            maxIndex = i;
        end
    end

    % Bounding box for the largest contour
    stats = regionprops(L == maxIndex, 'BoundingBox');
    if isempty(stats)
        continue;  % Skip if no region found
    end
    bbox = round(stats.BoundingBox);  % [x, y, width, height]
    x = bbox(1);
    y = bbox(2);
    w = bbox(3);
    h = bbox(4);

    % Ensure bounding box is within image bounds
    xEnd = min(x + w - 1, size(img, 2));
    yEnd = min(y + h - 1, size(img, 1));
    x = max(1, x);
    y = max(1, y);

    % Crop ROI
    roi = img(y:yEnd, x:xEnd, :);

    % Resize to 1024x1024
    resized_roi = imresize(roi, [1024 1024]);

    % Save cropped image
    outPath = fullfile(outputDir, imgName);
    imwrite(resized_roi, outPath);
end

disp('Batch cropping complete. Output saved to:');
disp(outputDir);
