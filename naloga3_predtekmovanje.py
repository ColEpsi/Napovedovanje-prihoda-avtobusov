import gzip, csv, linear, lpputils, datetime
import numpy as np

def absolute_error(cas, napoved):
    if cas == "?": return 0
    return abs(lpputils.tsdiff(cas, napoved))

def napolni_x(d, prazniki):
    primer = []
    datum = lpputils.parsedate(d[-3])
    primer.append(1 if datum.minute <= 30 else 0)
    primer.append(1 if datum.minute > 30 else 0)
    primer.append(1 if datum.hour < 20 and datum.hour > 6 else 0)
    for ura in range(24): primer.append(1 if datum.hour == ura else 0)
    for dan in range(7): primer.append(1 if datum.weekday() == dan else 0)
    primer.append(0)
    for praznik in prazniki:
        if datum.date() == praznik:
            primer[len(primer) - 1] = 1
            break
    return primer

def zgradi_matrike(ime_datoteke, training):
    d = open("prazniki_in_dela_prosti_dnevi.csv", "rt", encoding="latin1")
    branje = csv.reader(d)
    next(branje)
    prazniki = []
    for d in branje:
        prazniki.append(datetime.datetime.strptime((d[0].split(";", 1))[0], "%d.%m.%Y").date())

    f = gzip.open(ime_datoteke, "rt", encoding="latin1")
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    if training:
        x = []
        y = []
        for d in reader:
            x.append(napolni_x(d, prazniki))
            y.append(lpputils.tsdiff(lpputils.parsedate(d[-3]), lpputils.parsedate(d[-1])))
        X = linear.append_ones(np.array(x))
        Y = np.array(y)
        return X , Y
    else:
        x = []
        originalen_datum = []
        dejanski_cas = []
        for d in reader:
            originalen_datum.append(d[-3])
            dejanski_cas.append(d[-1])
            x.append(napolni_x(d, prazniki))
        X = linear.append_ones(np.array(x))
        return dejanski_cas, originalen_datum, X

X, Y = zgradi_matrike("train_pred.csv.gz", True)

linearna_regresija = linear.LinearLearner()
predikcijski_model = linearna_regresija(X, Y)

dejanski_cas, primeri, testni_X = zgradi_matrike("test_pred.csv.gz", False) #za izpis MAE spremeni v "train_pred.csv.gz"
mae_mesec = 11 #MAE za november
mae = 0
stevilo_primerov = 0
datoteka = open("napovedi.txt", "wt", encoding="latin1")
for vrstica in range(len(primeri)):
    napoved = lpputils.tsadd(primeri[vrstica], -predikcijski_model(testni_X[vrstica]))
    datoteka.write(napoved + "\n")
    if lpputils.parsedate(primeri[vrstica]).month == mae_mesec:
        mae += absolute_error(dejanski_cas[vrstica], napoved)
        stevilo_primerov += 1
datoteka.close()
if mae != 0: print("Mean absolute error:", mae / stevilo_primerov)

