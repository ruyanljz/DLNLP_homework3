import random
import re
import jieba
from gensim import corpora, models
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def data_processing(file_path, flag, count, length):
    """
    获取文件信息并预处理
    :param file_path: 文件名对应路径
    :param flag: 选择词/字为单位，0=词，1=字
    :param count: 一本小说所分成的段落数
    :param length: 每个段落的词个数
    :return data_out: 字符串形式的语料库
    :return words_out: 分词
    :return paragraph_out: 随机选取的段落
    """
    # 读取小说
    delete_symbol = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~「」『』（）]+'
    with open(file_path, 'r', encoding='ANSI') as f:
        data_out = f.read()
        data_out = data_out.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '')
        data_out = re.sub(delete_symbol, '', data_out)
        data_out = data_out.replace('\n', '')
        data_out = data_out.replace('\u3000', '')
        data_out = data_out.replace(' ', '')
        f.close()
    # 读取并删除停词
    with open('./cn_stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = []
        for a in f:
            if a != '\n':
                stopwords.append(a.strip())
    for a in stopwords:
        data_out = data_out.replace(a, '')
    # 以字或词为单位进行分词
    words_out = []
    if flag == 0:
        words_out = list(jieba.cut(data_out))
    elif flag == 1:
        words_out = [c for c in data_out]
    # 将小说中的词分为count个段落
    paragraph_out = []
    for ii in range(count):
        begin = random.randint(0, len(words_out)-length-1)
        paragraph_out.append(words_out[begin:begin+length])

    return data_out, words_out, paragraph_out


def lda_rf(paragraph_in, label_in, topics_in):
    """
    获取文件信
    :param paragraph_in: 200个段落
    :param label_in: 200个段落对应的标签
    :param topics_in: 主题数
    :return coherence_score_out: LDA的一致性
    :return accuracy_out: 随机森林分类的准确度
    :return topic_words_out: 五个主题中前10个特征词
    """
    # 将每个段落表示为词袋模型
    dictionary = corpora.Dictionary(paragraph_in)
    corpus_bow = [dictionary.doc2bow(word) for word in paragraph_in]
    # 训练LDA模型
    lda_model = models.LdaModel(corpus_bow, num_topics=topics_in, id2word=dictionary)
    # 将每个段落表示为主题分布
    corpus_lda = [lda_model[doc] for doc in corpus_bow]
    topic_probabilities = []
    for doc in corpus_lda:
        topic_temp = [0] * topics_in
        for topic, weight in doc:
            topic_temp[topic] = weight
        total_weight = sum(topic_temp)
        topic_probabilities.append([weight / total_weight for weight in topic_temp])
    # 计算一致性得分
    cm = models.CoherenceModel(model=lda_model, texts=paragraph_in, dictionary=dictionary, coherence='c_v')
    coherence_score_out = cm.get_coherence()
    # 取五个主题中前10个特征词
    topic_words_out = []
    for t in range(5):
        topic_word = lda_model.show_topic(t, topn=10)
        topic_words_out.append([word for word, _ in topic_word])

    # 划分训练集和测试集
    x = topic_probabilities
    y = label_in
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 训练随机森林
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    # print(classification_report(y_test, y_pred))
    accuracy_out = metrics.accuracy_score(y_test, y_pred)

    return coherence_score_out, accuracy_out, topic_words_out


if __name__ == "__main__":
    files = ['./data_novel/碧血剑.txt',
             './data_novel/飞狐外传.txt',
             './data_novel/鹿鼎记.txt',
             './data_novel/射雕英雄传.txt',
             './data_novel/神雕侠侣.txt',
             './data_novel/书剑恩仇录.txt',
             './data_novel/天龙八部.txt',
             './data_novel/侠客行.txt',
             './data_novel/笑傲江湖.txt',
             './data_novel/倚天屠龙记.txt']
    files_inf = ["碧血剑", "飞狐外传", "鹿鼎记", "射雕英雄传", "神雕侠侣", "书剑恩仇录", "天龙八部", "侠客行", "笑傲江湖", "倚天屠龙记"]

    para = 20
    word_num = 600
    # **********以 词 为基本单元分类**********
    paragraph_ci = []
    label_ci = []
    for i, file in enumerate(files):
        data, words, paragraph = data_processing(file, 0, para, word_num)  # 一共10本小说，每一本分成20段，每一段600个词
        paragraph_ci.extend(paragraph)
        label_ci.extend([i] * para)
    print("词 文件读取完成")
    coherence_score_ci = []
    accuracy_ci = []
    topic_words_ci = []
    for topics in range(10, 410, 20):
        coherence_score, accuracy, topic_words = lda_rf(paragraph_ci, label_ci, topics)
        coherence_score_ci.append(coherence_score)
        accuracy_ci.append(accuracy)
        topic_words_ci.append(topic_words)
    print("词分类结束，其中五个主题下前10个特征词分别为：")
    for i in range(5):
        print("主题"+str(i), ":", topic_words_ci[1][i])

    # **********以 字 为基本单元分类**********
    paragraph_zi = []
    label_zi = []
    for i, file in enumerate(files):
        data, words, paragraph = data_processing(file, 1, para, word_num)  # 一共10本小说，每一本分成20段，每一段600个词
        paragraph_zi.extend(paragraph)
        label_zi.extend([i] * para)
    print("字 文件读取完成")
    coherence_score_zi = []
    accuracy_zi = []
    topic_words_zi = []
    for topics in range(10, 410, 20):
        coherence_score, accuracy, topic_words = lda_rf(paragraph_zi, label_zi, topics)
        coherence_score_zi.append(coherence_score)
        accuracy_zi.append(accuracy)
        topic_words_zi.append(topic_words)
    print("字分类结束，其中五个主题下前10个特征词分别为：")
    for i in range(5):
        print("主题"+str(i), ":", topic_words_zi[1][i])

    # 画图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
    plt.rcParams['axes.unicode_minus'] = False
    x = range(10, 410, 20)
    plt.figure(1)
    plt.plot(x, coherence_score_ci, 'r-', alpha=0.8, linewidth=1, label='ci')
    plt.plot(x, coherence_score_zi, 'b-', alpha=0.8, linewidth=1, label='zi')
    plt.legend(loc='best')
    plt.xlabel('主题数', fontdict={'size': 14})
    plt.ylabel('LDA一致性', fontdict={'size': 14})
    plt.title("不同主题数下LDA模型一致性", fontdict={'size': 16})

    plt.figure(2)
    plt.plot(x, accuracy_ci, 'r-', alpha=0.8, linewidth=1, label='ci')
    plt.plot(x, accuracy_zi, 'b-', alpha=0.8, linewidth=1, label='zi')
    plt.legend(loc='best')
    plt.xlabel('主题数', fontdict={'size': 14})
    plt.ylabel('RF分类准确性', fontdict={'size': 14})
    plt.title("不同主题数下RF分类准确性", fontdict={'size': 16})
    plt.show()
