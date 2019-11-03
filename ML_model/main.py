#import ToxicCommentPredictor
from ToxicCommentPredictor import ToxicCommentPredictor

toxic = ToxicCommentPredictor()
#toxic = ToxicCommentPredictor.ToxicCommentPredictor()
rnn_y = toxic.rnn_prediction('cocksucker before you piss around on my work')
print(rnn_y)
cnn_y = toxic.cnn_prediction('cocksucker before you piss around on my work')
print(cnn_y)

# result =  ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
