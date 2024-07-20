from datasetpre.process.csvtofasta import fromcsvtofasta_win

taglist=['AGC','Atypical','CAMK','CDK','CK2','CMGC','MAPK','PKA','PKC','Src','TK','ST','Y']
for tag in taglist:
    root = f"D:\lujiale\PtmDeep\deep\data\dataset\cross_dataset_new\{tag}/2000/"
    train_set=root+"train_set1.csv"
    posout=root+"pos.fasta"
    test_set = root + "test_set1.csv"
    negout=root+"neg.fasta"
    fromcsvtofasta_win(train_set,posout,negout)
    fromcsvtofasta_win(test_set, posout,negout)
    print(f"done:{tag}")