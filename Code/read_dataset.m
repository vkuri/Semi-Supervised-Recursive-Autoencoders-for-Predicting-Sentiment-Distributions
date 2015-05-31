function [] = read_dataset(params_dataset, parameters)
% Function for reading the dataset

% Making use of vocabulary list from Socher's paper
load(strcat(params_dataset.path,'vocab.mat'), 'words');

% Randomly initialize each word in the vocabulary list with a weight
interval = parameters.distribution_interval;
ww = (rand([parameters.word_size size(words,2)]) * 2 * interval) - interval;

% Read the positive and negative reviews file
file_pos = strcat(params_dataset.path, params_dataset.filename_positive);
file_neg = strcat(params_dataset.path, params_dataset.filename_negative);
f1 = fopen(file_pos);
f2 = fopen(file_neg);

% Loading the entire dictionary with weights for each word
file_bin_pos = strcat(params_dataset.path, params_dataset.pos_binarized);
file_bin_neg = strcat(params_dataset.path, params_dataset.neg_binarized);
load(file_bin_pos, 'allSNum', 'allSStr');
dictNum_pos = allSNum;
dictStr_pos = allSStr;
load(file_bin_neg, 'allSNum', 'allSStr');
dictNum = [dictNum_pos allSNum];
dictStr = [dictStr_pos allSStr];

% Adapted from Socher's code to perform manual changes to the dictionary
dictNum{5919} = 4722;
dictStr{5919} = {'obvious'};

dictNum{6550} = 20144;
dictStr{6550} = {'horrible'};

dictNum{9801} = 241212;
dictStr{9801} = {'crummy'};

% Label each positive review by 1 and negative review by 0.
labels = zeros(1,10^5);
sentence_words = cell(1,10^5);
ctr = 0;
while ~feof(f1)
    entry = fgetl(f1);
    while strcmp(entry(end),' ')||strcmp(entry(end),'.')
        entry(end) = [];
    end
    ctr = ctr + 1;
    sentence_words{ctr} = regexp(entry,' ','split');
    labels(ctr) = 1;
end

while ~feof(f2)
    entry = fgetl(f2);
    while strcmp(entry(end),' ')||strcmp(entry(end),'.')
        entry(end) = [];
    end
    ctr = ctr + 1;
    sentence_words{ctr} = regexp(entry,' ','split');
    labels(ctr) = 0;
end

sentence_words(ctr+1:end) = [];
labels(ctr+1:end) = [];
num_reviews = ctr;

% Converting list of words to a wordMap to change some words, adapted from
% Socher's code.

wordMap = containers.Map(words,1:length(words));

wordMap('elipsiseelliippssiiss') = wordMap('...');
wordMap('smilessmmiillee') = (uint32(wordMap.Count) + 1);   %end-4
ww = [ww ww(:,wordMap('smile'))];
wordMap('frownffrroowwnn') = (uint32(wordMap.Count) + 1);   %end-3
ww = [ww ww(:,wordMap('frown'))];
wordMap('haha') = (uint32(wordMap.Count) + 1);      %end-2
ww = [ww ww(:,wordMap('laugh'))];
wordMap('hahaha') = (uint32(wordMap.Count));        %end-2
ww = [ww ww(:,wordMap('laugh'))];
wordMap('hahahaha') = (uint32(wordMap.Count));      %end-2
ww = [ww ww(:,wordMap('laugh'))];
wordMap('hahahahaha') = (uint32(wordMap.Count));    %end-2
ww = [ww ww(:,wordMap('laugh'))];
wordMap('hehe') = (uint32(wordMap.Count) + 1);      %end-1
ww = [ww ww(:,wordMap('laugh'))];
wordMap('hehehe') = (uint32(wordMap.Count));        %end-1
ww = [ww ww(:,wordMap('laugh'))];
wordMap('hehehehe') = (uint32(wordMap.Count));      %end-1
ww = [ww ww(:,wordMap('laugh'))];
wordMap('hehehehehe') = (uint32(wordMap.Count));    %end-1
ww = [ww ww(:,wordMap('laugh'))];
wordMap('lol') = (uint32(wordMap.Count) + 1);       %end
ww = [ww ww(:,wordMap('laugh'))];
wordMap('lolol') = (uint32(wordMap.Count));         %end
ww = [ww ww(:,wordMap('laugh'))];
words = [words {'elipsiseelliippssiiss'} {'smilessmmiillee'} {'frownffrroowwnn'} {'haha'} {'hahaha'} {'hahahaha'} {'hahahahaha'} {'hehe'} {'hehehe'} {'hehehehe'} {'hehehehehe'} {'lol'} {'lolol'}];

words_indexed = cell(num_reviews,1);
words_reIndexed = cell(num_reviews,1);

words_embedded = cell(num_reviews,1);
sentence_length = cell(num_reviews,1);

for i=1:num_reviews
    words_indexed{i} = dictNum{i};
    words_embedded{i} = ww(:,words_indexed{i});
    sentence_length{i} = length(words_indexed{i});
end

index_list = cell2mat(words_indexed');
unq = sort(index_list);
freq = histc(index_list,unq);
unq(freq==0) = [];
freq(freq==0) = [];

reIndexMap = containers.Map(unq,1:length(unq));
words2 = words(unq);

parfor i=1:num_reviews
    words_reIndexed{i} = arrayfun(@(x) reIndexMap(x), words_indexed{i});
end

ww = ww(:, unq);

% K fold partitions
cv_obj_path = strcat(params_dataset.path, params_dataset.cv_obj);
load(cv_obj_path);

full_train_ind = cv_obj.training(params_dataset.kfold);
full_train_nums = find(full_train_ind);
test_ind = cv_obj.test(params_dataset.kfold);
test_nums = find(test_ind);

train_ind = full_train_ind;
cv_ind = test_ind;

dictNum = words_reIndexed;

isnonZero = ones(1,length(dictNum));

preProFile_path = strcat(params_dataset.path, params_dataset.filename_preprofile);
save(preProFile_path, 'labels', 'words_reIndexed', 'full_train_ind','train_ind','cv_ind','test_ind','ww','dictNum','unq','isnonZero','test_nums','full_train_nums');

end