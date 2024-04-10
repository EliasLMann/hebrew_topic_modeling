import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess

# Preprocess text data
def preprocess_text(text):
    return [token for token in simple_preprocess(text) if token not in gensim.parsing.preprocessing.STOPWORDS]

def load_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

# # File path to your text file
file_path = 'C:\\Users\\tanay\OneDrive\\Desktop\\genesis_hebrew.txt'

# # Load data from file
#data = load_data_from_file(file_path)

from multiprocessing import process
# Load data
data = ["רֵאשִׁית בּרא אֱלֹהִים שָׁמַיִם אֶרֶץ אֶרֶץ היה תֹּהוּ בֹּהוּ חֹשֶׁךְ פָּנֶה תְּהוֺם רוּחַ אֱלֹהִים רחף פָּנֶה מַיִם אמר אֱלֹהִים היה אוֺר היה אוֺר ראה אֱלֹהִים אוֺר כִּי טוֺב בּדל אֱלֹהִים בַּיִן אוֺר בַּיִן חֹשֶׁךְ קרא אֱלֹהִים אוֺר יוֺם חֹשֶׁךְ קרא לַיְלָה היה עֶרֶב היה בֹּקֶר יוֺם אֶחָד ף אמר אֱלֹהִים היה רָקִיעַ תָּוֶךְ מַיִם היה בּדל בַּיִן מַיִם מַיִם עשׂה אֱלֹהִים רָקִיעַ בּדל בַּיִן מַיִם אֲשֶׁר תַּחַת רָקִיעַ בַּיִן מַיִם אֲשֶׁר רָקִיעַ היה כֵּן קרא אֱלֹהִים רָקִיעַ שָׁמַיִם היה עֶרֶב היה בֹּקֶר יוֺם שֵׁנִי ף אמר אֱלֹהִים קוה מַיִם תַּחַת שָׁמַיִם מָקוֺם אֶחָד ראה יַבָּשָׁה היה כֵּן קרא אֱלֹהִים יַבָּשָׁה אֶרֶץ מִקְוֶה מַיִם קרא יָם ראה אֱלֹהִים כִּי טוֺב אמר אֱלֹהִים דּשׁא אֶרֶץ דֶּשֶׁא עֵשֶׂב זרע זֶרַע עֵץ פְּרִי עשׂה פְּרִי מִין אֲשֶׁר זֶרַע בּ אֶרֶץ היה כֵּן יצא אֶרֶץ דֶּשֶׁא עֵשֶׂב זרע זֶרַע מִין עֵץ עשׂה פְּרִי אֲשֶׁר זֶרַע בּ מִין ראה אֱלֹהִים כִּי טוֺב היה עֶרֶב היה בֹּקֶר יוֺם שְׁלִישִׁי ף אמר אֱלֹהִים היה מָאוֺר רָקִיעַ שָׁמַיִם בּדל בַּיִן יוֺם בַּיִן לַיְלָה היה אוֺת מוֺעֵד יוֺם שָׁנָה היה מָאוֺר רָקִיעַ שָׁמַיִם אור אֶרֶץ היה כֵּן עשׂה אֱלֹהִים שְׁנַיִם מָאוֺר גָּדוֺל מָאוֺר גָּדוֺל מֶמְשָׁלָה יוֺם מָאוֺר קָטֹן מֶמְשָׁלָה לַיְלָה כּוֺכָב נתן אֱלֹהִים רָקִיעַ שָׁמַיִם אור אֶרֶץ משׁל יוֺם לַיְלָה בּדל בַּיִן אוֺר בַּיִן חֹשֶׁךְ ראה אֱלֹהִים כִּי טוֺב היה עֶרֶב היה בֹּקֶר יוֺם רְבִיעִי ף אמר אֱלֹהִים שׁרץ מַיִם שֶׁרֶץ נֶפֶשׁ חַי עוֺף עוף אֶרֶץ פָּנֶה רָקִיעַ שָׁמַיִם בּרא אֱלֹהִים תַּנִּין גָּדוֺל כֹּל נֶפֶשׁ חַי רמשׂ אֲשֶׁר שׁרץ מַיִם מִין כֹּל עוֺף כָּנָף מִין ראה אֱלֹהִים כִּי טוֺב בּרךְ אֱלֹהִים אמר פּרה רבה מלא מַיִם יָם עוֺף רבה אֶרֶץ היה עֶרֶב היה בֹּקֶר יוֺם חֲמִישִׁי ף אמר אֱלֹהִים יצא אֶרֶץ נֶפֶשׁ חַי מִין בְּהֵמָה רֶמֶשׂ חַיָּה אֶרֶץ מִין היה כֵּן עשׂה אֱלֹהִים חַיָּה אֶרֶץ מִין בְּהֵמָה מִין כֹּל רֶמֶשׂ אֲדָמָה מִין ראה אֱלֹהִים כִּי טוֺב אמר אֱלֹהִים עשׂה אָדָם צֶלֶם דְּמוּת רדה דָּגָה יָם עוֺף שָׁמַיִם בְּהֵמָה כֹּל אֶרֶץ כֹּל רֶמֶשׂ רמשׂ אֶרֶץ בּרא אֱלֹהִים אָדָם צֶלֶם צֶלֶם אֱלֹהִים בּרא זָכָר נְקֵבָה בּרא בּרךְ אֱלֹהִים אמר אֱלֹהִים פּרה רבה מלא אֶרֶץ כּבשׁ רדה דָּגָה יָם עוֺף שָׁמַיִם כֹּל חַיָּה רמשׂ אֶרֶץ אמר אֱלֹהִים הִנֵּה נתן כֹּל עֵשֶׂב זרע זֶרַע אֲשֶׁר פָּנֶה כֹּל אֶרֶץ כֹּל עֵץ אֲשֶׁר בּ פְּרִי עֵץ זרע זֶרַע היה אָכְלָה כֹּל חַיָּה אֶרֶץ כֹּל עוֺף שָׁמַיִם כֹּל רמשׂ אֶרֶץ אֲשֶׁר בּ נֶפֶשׁ חַי כֹּל יֶרֶק עֵשֶׂב אָכְלָה היה כֵּן ראה אֱלֹהִים כֹּל אֲשֶׁר עשׂה הִנֵּה טוֺב מְאֹד היה עֶרֶב היה בֹּקֶר יוֺם שִׁשִּׁי ף"]
processed_data= [preprocess_text(text) for text in data]

# Create dictionary and corpus
dictionary = corpora.Dictionary(processed_data)
corpus = [dictionary.doc2bow(text) for text in processed_data]

# Build LDA model
lda_model = LdaModel(corpus=corpus,
                     id2word=dictionary,
                     num_topics=4,
                     random_state=42,
                     update_every=1,
                     chunksize=100,
                     passes=10,
                     alpha='auto',
                     per_word_topics=True)


# Print top 10 words for each topic
for topic in lda_model.print_topics(num_topics=10, num_words=5):
    print(topic)


# coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, texts=processed_data, dictionary=dictionary, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print(f"Coherence score: {coherence_lda}")