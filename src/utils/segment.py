import codecs
import os

class segment:
    def __init__(self):
        LTP_DATA_DIR = 'resources/ltp_data_v3.4.0/'
        cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
        
        from pyltp import Segmentor
        self.segmentor = Segmentor()
        self.segmentor.load_with_lexicon(cws_model_path, '/path/to/your/lexicon')
        
    def seg(self, text):
        words = self.segmentor.segment(text)
        return words

    def destroy(self):
        self.segmentor.release()
        
    def segFile(self, infile, outfile):
        data = codecs.open(infile, 'r')
        out = codecs.open(outfile, 'w')
        for line in data:
            fields = line.strip().split('\t')
            out.write(fields[0] + '\t' + '\t'.join([' '.join(self.seg(fields[i])) for i in range(1, len(fields))]) + '\n')
        data.close()
        out.close()

if __name__=="__main__":
    segtest = segment()
    p = 'data/intent/small/'
    segtest.segFile(p + 'intent.train', p + 'intent-seg.train')
    import os.path
    if os.path.exists(p + 'intent.test'):
        segtest.segFile(p + 'intent.test', p + 'intent-seg.test')
    segtest.destroy()
