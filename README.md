
# bert crf
## 数据预处理
* preprocess/preprocess_main.py
## 模型训练与保存

使用transformers预训练

--train_file   训练数据集

--dev_file   验证数据集

--weight_dir   模型权重存储

--model_dir   模型存储

--n_vocab   词典大小  21862   bert原有21128

--n_label   标签个数

--seq_length  序列最大长度

--print_step

--epoch

--droprate

--batch_size    batch_size=64是最好的

--pretrain_name   预训练语言模型名称

--cache_dir    transformers预训练语言模型缓存路径


python bert_crf.py  --train_file  "./data/train.tfrecord"  \
                    --dev_file  "./data/dev.tfrecord"  \
                    --weight_dir  "./weights/BertCrfModel/BertCrfModel"  \
                    --model_dir  "./models/bert_crf"  \
                    --n_vocab  21862  \
                    --n_label  65  \
                    --seq_length  512  \
                    --print_step  5  \
                    --epoch  30  \
                    --droprate  0.2  \
                    --batch_size  64

## 模型测试

--test_file   测试的tf_record

python model_test.py  --test_file  "./data/test.tfrecord"  \
                      --test_res_file  "./data/test_res.txt"  \
                      --weight_dir  "./weights/BertCrfModel/BertCrfModel"  \
                      --model_dir  "./models/bert_crf"

## 其他

### 对比来结果和模型预测结果一同 

diff_compare.py

### 更改 vocab_size 大小
tf_bert_pretrained_model.py

重写  load_tf_weights方法  saved_weight_value
代码第129～143行

transformers 预训练

预训练模型继承关系：

TFBert(TFBertPreTrainedModel)

TFBertForPreTraining.TFBertPreTrainedModel(TFPreTrainedModel)

TFPreTrainedModel(tf.keras.Model, TFModelUtilsMixin, TFGenerationMixin, PushToHubMixin)

https://huggingface.co/transformers/_modules/transformers/modeling_tf_utils.html#TFPreTrainedModel

重写  load_tf_weights方法  saved_weight_value