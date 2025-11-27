# started 11/20/2025



class MVP:

    def __init__(self, ask_input=False):





        # import model, scalers, explainer
        self.scalers = joblib.load("scalers.pkl")
        self.xgb_model = joblib.load("xgb_model.pkl")
        with open("lime_explainer.dill", "rb") as f:
            bundle = dill.load(f)

        self.explainer = bundle["explainer"]
        self.predict_fn = bundle["predict_fn"]

        # setup scalers
        self.std_scaler = self.scalers["standard_scaler"]
        self.pwr_scaler = self.scalers["power_scaler"]
        self.power_cols = ["precip_in", "pop_density"]
        self.standard_cols = ["temp_max_F", "humidity_pct", "windspeed_mph", "ndvi","slope"]


        # get user input or set defaults
        self.input_vec = []
        self.ask_input = ask_input
        if self.ask_input:
            self.get_user_input()
        else:
            self.max_temp = 75
            self.humidity = 50
            self.windspeed_mph = 6
            self.precip = 0
            self.ndvi = 3000
            self.pop_den = 200
            self.slope = 600

        # create feature vector and dataframe
        self.input_vec = [[self.max_temp],
                          [self.humidity],
                          [self.precip],
                          [self.windspeed_mph],
                          [self.ndvi],
                          [self.pop_den],
                          [self.slope]]
        
        self.input_df = pd.DataFrame()
        self.input_df["temp_max_F"]=self.input_vec[0]
        self.input_df["humidity_pct"]=self.input_vec[1]
        self.input_df["precip_in"]=self.input_vec[2]
        self.input_df["windspeed_mph"]=self.input_vec[3]
        self.input_df["ndvi"]=self.input_vec[4]
        self.input_df["pop_density"]=self.input_vec[5]
        self.input_df["slope"]=self.input_vec[6]

        self.input_df_nonscaled = self.input_df.copy()


        # scale input data
        self.input_df[self.standard_cols] = self.std_scaler.transform(self.input_df[self.standard_cols])
        self.input_df[self.power_cols] = self.pwr_scaler.transform(self.input_df[self.power_cols])

        # setup instance for lime
        self.x_instance = np.array([self.input_df.iloc[0].to_list()])


    def get_user_input(self): # ask user for input values
        self.max_temp = float(input("enter max temp F: "))
        self.humidity = float(input("enter humidity pct: "))
        self.windspeed_mph = float(input("enter windspeed mph: "))
        self.precip = float(input("enter precip in: "))
        self.ndvi = float(input("enter ndvi: "))
        self.pop_den = float(input("enter pop den ppl/sq km: "))
        self.slope = float(input("enter ruggedness rise/run*100: "))



    def get_pred(self, acres=False): # get prediction from model
        
        predicted_fire_size = self.xgb_model.predict(self.input_df).tolist()[0]

        if acres: # if user wants raw prediction + acres
            return predicted_fire_size, 10**predicted_fire_size # undo log scale

        return predicted_fire_size
    



    def lime_graph(self): # generate lime explanation graph

        explanation = self.explainer.explain_instance(self.x_instance.flatten(), self.predict_fn)

        output, acres = self.get_pred(acres=True)




        explanation.as_pyplot_figure()

        input_text = "\n".join(f"{col}: {val}" for col, val in self.input_df_nonscaled.iloc[0].items())

        plt.text(
            x=plt.xlim()[1] + 0.01 * plt.xlim()[1],
            y=0.5 * plt.ylim()[1],
            s=input_text,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='left'
        )
        

        plt.subplots_adjust(right=0.75)
        plt.subplots_adjust(left=0.25)

        plt.title(label=f"explenation - predicted: {output:.2f}, in acres: {acres:.2f}")
        plt.show()




import pandas as pd
import joblib
import dill
import matplotlib.pyplot as plt
import numpy as np
#import xgboost
#import lime

mvp = MVP(ask_input=True)

pred, acres = mvp.get_pred(acres=True)
mvp.lime_graph()

print(f"raw model output: {pred:.2f}  in acres: {acres:.2f}")


# import xgboost, os, lime
# print(os.path.dirname(xgboost.__file__))
# print(os.path.dirname(lime.__file__))



## ranges (of input data used when creating model):
# max temp: 43-118
# humidity: 6-100
# windspeed: 2.5-20
# precip: 0-0.5
# ndvi: 0-6000 (but outlier at -2000)  
# pop den: 0-5000
# slope: 0-2500
## model probably can't extrapolate well, so stay in these ranges
