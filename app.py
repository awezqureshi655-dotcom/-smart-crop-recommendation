importstreamlitasst
importpandasaspd
importos
fromtabpfn_clientimportTabPFNClassifier

#--------------------------------------------------
#PAGESETTINGS
#--------------------------------------------------

st.set_page_config(
page_title="AISmartCropRecommendation",
page_icon="🌱",
layout="centered"
)

#--------------------------------------------------
#TABPFNAUTHENTICATION
#--------------------------------------------------

#StreamlitSecretscontainsTABPFN_TOKEN.
#MakesureitisavailabletoTabPFN-client.
os.environ["TABPFN_TOKEN"]=st.secrets["TABPFN_TOKEN"]

#--------------------------------------------------
#LOADDATA
#--------------------------------------------------

df=pd.read_csv("crop_data.csv")

features=[
"N",
"P",
"K",
"temperature",
"humidity",
"ph",
"rainfall"
]

#--------------------------------------------------
#APPTITLE
#--------------------------------------------------

st.title("🌱AI-BasedSmartCropRecommendationSystem")

st.write(
"Enterthesoilandweatherconditionsbelowtogetan"
"AI-basedcroprecommendationusingTabPFN."
)

st.divider()

#--------------------------------------------------
#INPUTS
#--------------------------------------------------

st.subheader("🌾Soil&WeatherConditions")

N=st.number_input(
"Nitrogen(N)",
min_value=0.0,
max_value=200.0,
value=90.0
)

P=st.number_input(
"Phosphorus(P)",
min_value=0.0,
max_value=200.0,
value=42.0
)

K=st.number_input(
"Potassium(K)",
min_value=0.0,
max_value=200.0,
value=43.0
)

temperature=st.number_input(
"Temperature(°C)",
min_value=-10.0,
max_value=60.0,
value=20.88
)

humidity=st.number_input(
"Humidity(%)",
min_value=0.0,
max_value=100.0,
value=82.0
)

ph=st.number_input(
"SoilpH",
min_value=0.0,
max_value=14.0,
value=6.5
)

rainfall=st.number_input(
"Rainfall(mm)",
min_value=0.0,
max_value=1000.0,
value=202.94
)

st.divider()

#--------------------------------------------------
#RECOMMENDATION
#--------------------------------------------------

if st.button("🌱 Recommend Crop", use_container_width=True):

withst.spinner("TabPFNAIisanalyzingtheconditions..."):

try:
#Trainingdata
X=df[features]
y=df["label"]

#CreateTabPFNmodel
model=TabPFNClassifier(
model_path="v3.5_default",
n_estimators=8
)

#TrainTabPFNusingthedataset
model.fit(X,y)

#Userinput
new_input=pd.DataFrame([{
"N":N,
"P":P,
"K":K,
"temperature":temperature,
"humidity":humidity,
"ph":ph,
"rainfall":rainfall
}])

#Prediction
prediction=model.predict(new_input)[0]

#Probabilities
probabilities=model.predict_proba(new_input)[0]
classes=model.classes_

#Top3
top3_indices=probabilities.argsort()[-3:][::-1]

st.success(
f"🌱RecommendedCrop:{prediction.upper()}"
)

st.subheader("📊Top3AIRecommendations")

forrank,indexinenumerate(top3_indices,start=1):

crop=classes[index]
probability=probabilities[index]*100

st.write(
f"**{rank}.{crop.upper()}—"
f"{probability:.2f}%modelconfidence**"
)

st.progress(
min(float(probability)/100,1.0)
)

st.divider()

#--------------------------------------------------
#CROPINFORMATION
#--------------------------------------------------

crop_rows=df[
df["label"].str.lower()==str(prediction).lower()
]

ifnotcrop_rows.empty:

crop_info=crop_rows.iloc[0]

st.subheader("🌾CropInformation")

col1,col2=st.columns(2)

withcol1:
st.write(
f"**Season:**{crop_info['season']}"
)
st.write(
f"**SoilType:**{crop_info['soil_type']}"
)

withcol2:
st.write(
f"**WaterNeed:**{crop_info['water_need']}"
)
st.write(
f"**State:**{crop_info['state']}"
)

st.caption(
"Note:Thedisplayedpercentagesrepresentthe"
"TabPFNmodel'sconfidencescores,notguaranteed"
"real-worldcropsuccessprobabilities."
)

exceptExceptionase:

st.error("TheTabPFNmodelcouldnotrun.")

st.write(
"Pleasetryagainaftertheapplicationfinishesloading."
)

st.code(str(e))
