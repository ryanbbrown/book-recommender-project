# book text reader
import os
import codecs

files = sorted([i for i in os.listdir('book_text/')])

booklist = []

for file in files:
    with codecs.open('book_text/'+file, 'r', encoding='utf8',) as book:
        booklist.append(book.read()[0:100000])
        
df.insert(3, 'text', booklist)


# book text splitter
df1, df2, df3, df4, df5 = df.copy(), df.copy(), df.copy(), df.copy(), df.copy()

dflist = [df1, df2, df3, df4, df5]
ranges = [0,20000,40000,60000,80000,100000]

for df, i in zip(dflist, range(1,7)):
    df['text'] = df['text'].apply(lambda x: x[ranges[i-1]:ranges[i]])
    
df = df1.append([df2, df3, df4, df5])
df.insert(4, 'section_number', [1]*40+[2]*40+[3]*40+[4]*40+[5]*40)
