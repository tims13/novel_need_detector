# for classifier
num_epochs = 10
train_test_ratio = 0.80
train_valid_ratio = 0.80
# for novelty
feature_len = 64
novel_test_num = 70
# folders
data_folder = 'data/'
des_folder = 'record/'

# original data
data_sentence_path = data_folder + 'annotation_sentence.xlsx'
data_need_csv_path = data_folder + 'need_all_sentence.csv'
data_need_simple_path = data_folder + 'need_simple_sentence.csv'
data_novel_path = data_folder + 'novel_sentence.xlsx'
# intermediate result for checking
data_pos_csv_path = data_folder + 'data_pos.csv'
data_neg_csv_path = data_folder + 'data_neg.csv'
data_irre_csv_path = data_folder + 'data_irre.csv'
data_np_path = data_folder + 'train_test'
data_train_csv_path = data_folder + 'train_novel.csv'
data_need_detected_path = data_folder + 'need_detected.csv'
data_novel_test_path = data_folder + 'novel_test.csv'
data_novel_train_path = data_folder + 'novel_train.csv'
# results
data_need_results_path = des_folder + 'need_results.csv'
data_result_csv_path = des_folder + 'novel_results.csv'