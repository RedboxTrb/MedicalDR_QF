clc; clear;

inputFolder = "D:\DREAM Dataset\1030The SUSTech-SYSU dataset for automated exudate detection and diabetic retinopathy grading (1)\croppedimages";
outputFolder = "D:\DREAM Dataset\1030The SUSTech-SYSU dataset for automated exudate detection and diabetic retinopathy grading (1)\goodimages";
outputCSV = "D:\DREAM Dataset\1030The SUSTech-SYSU dataset for automated exudate detection and diabetic retinopathy grading (1)\labels.csv";

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Supported image extensions
extensions = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff'};

% Collect all matching image files
imageFiles = [];
for i = 1:length(extensions)
    imageFiles = [imageFiles; dir(fullfile(inputFolder, extensions{i}))];
end

results = {};

% Fuzzy thresholds
brisque_thresh = 50;
niqe_thresh = 6;
entropy_thresh = 5;
piqe_thresh = 40;

fprintf("🔍 Starting quality evaluation on %d images...\n", length(imageFiles));

for i = 1:length(imageFiles)
    filename = imageFiles(i).name;
    filepath = fullfile(inputFolder, filename);

    try
        img = imread(filepath);
        if size(img, 3) == 3
            gray = rgb2gray(img);
        else
            gray = img;
        end

       % Resize to 512×512 and normalize to [0, 1]
        img = imresize(img, [512, 512]);
        img = im2double(img);


        % Compute quality metrics
        brisque_score = brisque(img);
        niqe_score    = niqe(img);
        piqe_score    = piqe(img);
        entropy_score = entropy(gray);

        % Apply fuzzy rules
        cond1 = (brisque_score < brisque_thresh);
        cond2 = (niqe_score < niqe_thresh);
        cond3 = (piqe_score < piqe_thresh);
        cond4 = (entropy_score > entropy_thresh);
        satisfied = cond1 + cond2 + cond3 + cond4;

        if satisfied >= 3
            quality_label = "good";
            imwrite(img, fullfile(outputFolder, filename));
        else
            quality_label = "poor";
        end

        results(end+1, :) = {filename, brisque_score, niqe_score, piqe_score, entropy_score, quality_label};

        fprintf('[%03d] %s | B=%.2f, N=%.2f, P=%.2f, E=%.2f → %s\n', ...
                i, filename, brisque_score, niqe_score, piqe_score, entropy_score, quality_label);

    catch ME
        warning(" Skipping file: %s (%s)", filename, ME.message);
    end
end

% Write to CSV
headers = {'filename', 'brisque', 'niqe', 'piqe', 'entropy', 'quality_label'};
T = cell2table(results, 'VariableNames', headers);
writetable(T, outputCSV);

fprintf('\n Quality metrics saved to: %s\n', outputCSV);
fprintf(' Good quality images saved in: %s\n', outputFolder);
