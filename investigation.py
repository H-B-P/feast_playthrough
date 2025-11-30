# -*- coding: utf-8 -*-

# %%get dat

import durkon as du
import pandas as pd
df = pd.read_csv("../data.csv")
print(df)

# %%rename all

df = df.rename({"Feast Quality": "Q"}, axis=1)

for c in df.columns:
    df = df.rename({c: c[0]}, axis=1)

# %%dupery

dupeDf = df[df.duplicated()]
s = [c for c in df.columns if "Q" not in c]
ndupeDf = df[df.duplicated(subset=s)]

print(len(dupeDf))
print(len(ndupeDf))

# So there IS noise! Damn.

# %%Check resp dist

for i in range(22):
    sDf = df[df["Q"] == i]
    print(i, len(sDf), len(sDf)/len(df))

print(df["Q"].mean())

# Normalish distribution centered on ~13; longertailed down than up.

# %%Check expl prevs

for c in s:
    print(c, sum(df[c]), sum(df[c])/len(df))

# R common, A uncommon, P maybe uncommon, all else approx equiv

# %%Check corrs
pd.set_option('display.max_columns', None)
print(df.corr())

# Weak, noisy corrs. R maybe doesn't like K and M? small dset tho
# In terms of performace, E and G are aces, A and H kings, B also good,
# P tolerable, rest harmful

# %%Total, TSweet, TSpicy

s = [c for c in df.columns if "Q" not in c and len(c) == 1]
# All +ve contributors per last section. Proof!
sweet = ["A", "E", "G", "H", "P"]
# Not only that but sweeter implies higher performance! (this could be noise)
spicy = ["C", "F", "S", "V"]
# No smoking guns suggest spicy real, this is 100% vibes
meaty = ["R", "K", "M"]
# Justified by anticorrs


df["Total"] = df[s].sum(axis=1)
df["TSweet"] = df[sweet].sum(axis=1)
df["TSpicy"] = df[spicy].sum(axis=1)
df["TMeaty"] = df[meaty].sum(axis=1)

for i in range(10):
    sDf = df[df["Total"] == i]
    print(i, len(sDf), len(sDf)/len(df), sDf["Q"].mean())
print("")
for i in range(7):
    sDf = df[df["TSweet"] == i]
    print(i, len(sDf), len(sDf)/len(df), sDf["Q"].mean())
print("")
for i in range(6):
    sDf = df[df["TSpicy"] == i]
    print(i, len(sDf), len(sDf)/len(df), sDf["Q"].mean())
print("")
for i in range(5):
    sDf = df[df["TMeaty"] == i]
    print(i, len(sDf), len(sDf)/len(df), sDf["Q"].mean())

# Eyeballing suggests optimal: 5-6 foods, 2 sweet, 1 spicy, 1 meaty.
# But what if that's wrong? Total overall corrs with subcats!

# %%Prep


model = du.wraps.prep_additive_model(
    df, "Q", s, ["Total", "TSweet", "TSpicy", "TMeaty"])

# manually normalize in case it matters
# model["conts"]["TSweet"]=[[0,0],[1,0],[2,0],[5,0]]

print(model)

# %%Train
model = du.wraps.train_normal_model(df, 'Q', 5000, 0.01, model)
print(model)

# %%Look
for c in s:
    print(c, model["cats"][c]['uniques'][1]-model["cats"][c]['uniques'][0])
# Current guess: A, E, B, R, F


# %%Check highbies with new insight

print(df[df["Q"] == 20])

# All 4 have exactly 2 Sweet and either 1 or 2 spicy. Meaty weirdly irrelevant!

print(df[df["Q"] == 19])

# Roughly what we'd expect?

# %%ihunt, you hunt, we all hunt for interxes


du.wraps.interxhunt_normal_model(df, 'Q', s, [
                                 "Total", "TSweet", "TSpicy", "TMeaty"], model, filename="suggestions")

# K X V	catcat	0.236280870825132
# K X TSpicy	catcont	0.208385771062703
# E X G	catcat	0.172525049314132

# Top 3 interxns suggest K is spicy after all . . .
# (it was the kebabs that were killer, not the kraken)
# . . . and that there's something special about E and G
# Rest looks like noise or important X important
# (Unexpectedly, no meat / applesauce interxes jump out!)


# %%redefine, refine, refit

s = [c for c in df.columns if "Q" not in c and len(c) == 1]
desserts = ["E", "G"]
sweet = ["A", "H", "P"]  # , "E", "G"]
spicy = ["C", "F", "S", "V", "K"]


df["Total"] = df[s].sum(axis=1)
df["TSweet"] = df[sweet].sum(axis=1)
df["TSpicy"] = df[spicy].sum(axis=1)
df["TDesserts"] = df[desserts].sum(axis=1)

for i in range(10):
    sDf = df[df["Total"] == i]
    print(i, len(sDf), len(sDf)/len(df), sDf["Q"].mean())
print("")
for i in range(5):
    sDf = df[df["TSweet"] == i]
    print(i, len(sDf), len(sDf)/len(df), sDf["Q"].mean())
print("")
for i in range(7):
    sDf = df[df["TSpicy"] == i]
    print(i, len(sDf), len(sDf)/len(df), sDf["Q"].mean())
print("")
for i in range(4):
    sDf = df[df["TDesserts"] == i]
    print(i, len(sDf), len(sDf)/len(df), sDf["Q"].mean())

# %%Prep

model = du.wraps.prep_additive_model(
    df, "Q", s, ["Total", "TSweet", "TSpicy", "TDesserts"])

print(model)
# Surprisingly sweet on the sweetness. What am I not seeing?

# %%Train
model = du.wraps.train_normal_model(df, 'Q', 5000, 0.01, model)
print(model)

# %%Look
for c in s:
    print(c, model["cats"][c]['uniques'][1]-model["cats"][c]['uniques'][0])

# %%Check highbies with new insight

print(df[df["Q"] == 20])
# All four at the top had 2 sweets, 2 spicies, and no dessert!?
# Could be something something high variance. Could be coincidence. Hm . . .
print(df[df["Q"] == 19])
# Unenlightening.

# %%Time to test on the training set!
df["PREDICTED"] = du.misc.predict(df, model)
du.metrics.get_MAE(df, "PREDICTED", "Q")
# MAE=1.940

# %%Maybe we're tracking TOTAL sweetness?

df["TSweetness"] = df["TSweet"]+2*df["TDesserts"]

for i in range(8):
    sDf = df[df["TSweetness"] == i]
    print(i, len(sDf), len(sDf)/len(df), sDf["Q"].mean())

# %% Prep

model = du.wraps.prep_additive_model(
    df, "Q", s, ["Total", "TSweetness", "TSpicy"])

print(model)

# %%Train
model = du.wraps.train_normal_model(df, 'Q', 5000, 0.01, model)
print(model)

# Surprisingly sweet on the sweetness. What am I not seeing?

# %%Look
for c in s:
    print(c, model["cats"][c]['uniques'][1]-model["cats"][c]['uniques'][0])

# %%Time to test on the training set!
df["PREDICTED"] = du.misc.predict(df, model)
du.metrics.get_MAE(df, "PREDICTED", "Q")
# MAE=1.846, greater accuracy with greater simplicity means I'm right

# (Original, edited to have Kraken as Spicy, has 1.9 despite extra parameter.)

# %%Anything else missing?

du.wraps.interxhunt_normal_model(df, 'Q', s, [
                                 "Total", "TSpicy", "TSweetness"], model, filename="suggestions2")

# R X TSpicy	catcont	0.117264691564898
# "Roc is Spicy or shares some traits with spicies"
# K X V	catcat	0.11541970665814
# "You got the Spice Level of K and V wrong"
# R X Total	catcont	0.0967971090319754
# "Roc is . . . super filling? super bland? super common?"
# K X R	catcat	0.0910670250072819
# "Roc shares the aforementioned trait with Kraken in particular"
# B X E	catcat	0.0876910956713917
# "You also got the sweetness level of the desserts wrong"
# ". . . and/or fit sweetness with dots in the wrong places"

# %%OK, what on earth is up with meats?
# R is superprevalent, has strong anticorrs with BCV, has SUPER strong anticorrs with K and M.
meaty = ["R", "K", "M", "B", "C", "V"]

df["TMeaty"] = df[meaty].sum(axis=1)

for i in range(7):
    sDf = df[df["TMeaty"] == i]
    print(i, len(sDf), len(sDf)/len(df), sDf["Q"].mean())

print(df[df["TMeaty"] == 0])

# Doesn't seem to be any trick or gimmick to it, people just like having at least one Meat Dish.
# So: Roc anticorrs with K and V, K and V are spicier than I thought . . .

# %%Once more into the breach

df["TSpiciness"] = df["TSpicy"]+df["K"]+df["V"]

for i in range(8):
    sDf = df[df["TSpiciness"] == i]
    print(i, len(sDf), len(sDf)/len(df), sDf["Q"].mean())

# %%Prep

model = du.wraps.prep_additive_model(
    df, "Q", s, ["Total", "TSweetness", "TSpiciness"])

print(model)

# %%Train
model = du.wraps.train_normal_model(df, 'Q', 5000, 0.01, model)
print(model)

# %%Look
for c in s:
    print(c, model["cats"][c]['uniques'][1]-model["cats"][c]['uniques'][0])

# %%Check highbies with new insight

print(df[df["Q"] == 20])
print(df[df["Q"] == 19])
# Unenlightening.

# %%Time to test on the training set!
df["PREDICTED"] = du.misc.predict(df, model)
du.metrics.get_MAE(df, "PREDICTED", "Q")
# MAE=1.802

# %%Okay, what did I get wrong *this* time?

du.wraps.interxhunt_normal_model(df, 'Q', s, [
                                 "Total", "TSpicy", "TSweetness"], model, filename="suggestions3")

# R X TSpicy	catcont	0.113979123687168
# K X R	catcat	0.105579396928418
# R X Total	catcont	0.104511966784974
# . . . is Roc actually Spicy???

# %%Surely not . . .

df["TSpiciness"] = df["TSpicy"]+df["K"]+df["V"]+df["R"]

for i in range(8):
    sDf = df[df["TSpiciness"] == i]
    print(i, len(sDf), len(sDf)/len(df), sDf["Q"].mean())

model = du.wraps.prep_additive_model(
    df, "Q", s, ["Total", "TSweetness", "TSpiciness"])

print(model)

model = du.wraps.train_normal_model(df, 'Q', 5000, 0.01, model)
print(model)

df["PREDICTED"] = du.misc.predict(df, model)
du.metrics.get_MAE(df, "PREDICTED", "Q")

# Yeah it's worse. Correlations, leeching of imperfect linkages . . .
# . . . time to call it a day if we're down to that?

# %% Not quite done . . .


model = du.wraps.prep_additive_model(
    df, "Q", s, ["Total", "TSweetness", "TSpiciness"])

model = du.prep.add_catcat_to_model(model, df, "K", "R", replace=True)

model = du.wraps.train_normal_model(df, 'Q', 5000, 0.01, model)  # ,
# staticFeats = [c for c in model["cats"]] +
#            [c for c in model["conts"]])

print(model)
df["PREDICTED"] = du.misc.predict(df, model)
du.metrics.get_MAE(df, "PREDICTED", "Q")
# So what I see from this is K and R are super redundant. Hm.

# %%Look, once more

for c in model['cats']:
    print(c, model["cats"][c]['uniques'][1]-model["cats"][c]['uniques'][0])

# %%What rises to the top?

du.wraps.interxhunt_normal_model(df, 'Q', s, [
                                 "Total", "TSpiciness", "TSweetness"], model, filename="suggestions4")

# Ok yeah we're done here
