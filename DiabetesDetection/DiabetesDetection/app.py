import numpy as np
from flask import Flask, jsonify,render_template, request
import pickle

app = Flask(__name__)
lr_model = pickle.load(open('lr_model.pkl','rb'))
knn_model = pickle.load(open('knn_model.pkl','rb'))
gnb_model = pickle.load(open('gnb_model.pkl','rb'))
# model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('mainPage.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        Age = request.form.get('age')
        BMI = request.form.get('BMI')
        Fasting_gulcose = request.form.get('Fasting')
        A1C = request.form.get('A1C')

        arr = np.array([Age,BMI,Fasting_gulcose,A1C])

        arr_reshape = arr.reshape(1,-1)
        arr_reshape = arr_reshape.astype('float64')
        print(arr_reshape)

        lr_pred = lr_model.predict(arr_reshape)
        knn_pred = knn_model.predict(arr_reshape)
        gnb_pred = gnb_model.predict(arr_reshape)
        pred = [lr_pred,knn_pred,gnb_pred]
        m = -1
        i=0
        for j in range(len(pred)):
   
        # if counter `i` becomes 0
           if i == 0:
 
            # set the current candidate to `A[j]`
            m = pred[j]
 
            # reset the counter to 1
            i = 1
 
           elif m == pred[j]:
            i = i + 1
 
        # otherwise, decrement the counter if `A[j]` is a current candidate
           else:
            i = i - 1  

        final_pred = m
        if (lr_pred == 0):
            if (knn_pred == 0):
                if (gnb_pred == 0):
                    if (final_pred==0):
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="Healthy",gnb_pred_text="Healthy",pred_text="Healthy")
                    elif (final_pred==1):
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="Healthy",gnb_pred_text="Healthy",pred_text="Prediabetic")
                    else:
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="Healthy",gnb_pred_text="Healthy",pred_text="diabetic")
                elif(gnb_pred == 1):
                    if (final_pred==0):
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="Healthy",gnb_pred_text="Prediabetic",pred_text="Healthy")
                    elif (final_pred==1):
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="Healthy",gnb_pred_text="Prediabetic",pred_text="Prediabetic")
                    else:
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="Healthy",gnb_pred_text="Prediabetic",pred_text="diabetic")
                  
                else: 
                    if (final_pred==0):
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="Healthy",gnb_pred_text="diabetic",pred_text="Healthy")
                    elif (final_pred==1):
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="Healthy",gnb_pred_text="diabetic",pred_text="Prediabetic")
                    else:
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="Healthy",gnb_pred_text="diabetic",pred_text="diabetic")
                  
            elif (knn_pred==1):
                if (gnb_pred == 0):
                    if (final_pred==0):
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="Prediabetic",gnb_pred_text="Healthy",pred_text="Healthy")
                    elif (final_pred==1):
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="Prediabetic",gnb_pred_text="Healthy",pred_text="Prediabetic")
                    else:
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="Prediabetic",gnb_pred_text="Healthy",pred_text="diabetic")
                elif(gnb_pred == 1):
                    if (final_pred==0):
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="Prediabetic",gnb_pred_text="Prediabetic",pred_text="Healthy")
                    elif (final_pred==1):
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="Prediabetic",gnb_pred_text="Prediabetic",pred_text="Prediabetic")
                    else:
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="Prediabetic",gnb_pred_text="Prediabetic",pred_text="diabetic")
                  
                else: 
                    if (final_pred==0):
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="Prediabetic",gnb_pred_text="diabetic",pred_text="Healthy")
                    elif (final_pred==1):
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="Prediabetic",gnb_pred_text="diabetic",pred_text="Prediabetic")
                    else:
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="Prediabetic",gnb_pred_text="diabetic",pred_text="diabetic")
            else:
                 if (gnb_pred == 0):
                    if (final_pred==0):
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="diabetic",gnb_pred_text="Healthy",pred_text="Healthy")
                    elif (final_pred==1):
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="diabetic",gnb_pred_text="Healthy",pred_text="Prediabetic")
                    else:
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="diabetic",gnb_pred_text="Healthy",pred_text="diabetic")
                 elif(gnb_pred == 1):
                    if (final_pred==0):
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="diabetic",gnb_pred_text="Prediabetic",pred_text="Healthy")
                    elif (final_pred==1):
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="diabetic",gnb_pred_text="Prediabetic",pred_text="Prediabetic")
                    else:
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="diabetic",gnb_pred_text="Prediabetic",pred_text="diabetic")
                  
                 else: 
                    if (final_pred==0):
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="diabetic",gnb_pred_text="diabetic",pred_text="Healthy")
                    elif (final_pred==1):
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="diabetic",gnb_pred_text="diabetic",pred_text="Prediabetic")
                    else:
                        return render_template('mainPage.html', lr_pred_text="Healthy",knn_pred_text="diabetic",gnb_pred_text="diabetic",pred_text="diabetic")
        elif(lr_pred==1):
                if (knn_pred == 0):
                    if (gnb_pred == 0):
                        if (final_pred==0):
                          return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="Healthy",gnb_pred_text="Healthy",pred_text="Healthy")
                        elif (final_pred==1):
                         return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="Healthy",gnb_pred_text="Healthy",pred_text="Prediabetic")
                        else:
                         return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="Healthy",gnb_pred_text="Healthy",pred_text="diabetic")
                    elif(gnb_pred == 1):
                        if (final_pred==0):
                            return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="Healthy",gnb_pred_text="Prediabetic",pred_text="Healthy")
                        elif (final_pred==1):
                            return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="Healthy",gnb_pred_text="Prediabetic",pred_text="Prediabetic")
                        else:
                            return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="Healthy",gnb_pred_text="Prediabetic",pred_text="diabetic")
                    else: 
                        if (final_pred==0):
                          return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="Healthy",gnb_pred_text="diabetic",pred_text="Healthy")
                        elif (final_pred==1):
                          return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="Healthy",gnb_pred_text="diabetic",pred_text="Prediabetic")
                        else:
                          return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="Healthy",gnb_pred_text="diabetic",pred_text="diabetic")
                  
                elif (knn_pred==1):
                    if (gnb_pred == 0):
                        if (final_pred==0):
                          return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="Prediabetic",gnb_pred_text="Healthy",pred_text="Healthy")
                        elif (final_pred==1):
                          return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="Prediabetic",gnb_pred_text="Healthy",pred_text="Prediabetic")
                        else:
                          return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="Prediabetic",gnb_pred_text="Healthy",pred_text="diabetic")
                    elif(gnb_pred == 1):
                        if (final_pred==0):
                          return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="Prediabetic",gnb_pred_text="Prediabetic",pred_text="Healthy")
                        elif (final_pred==1):
                          return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="Prediabetic",gnb_pred_text="Prediabetic",pred_text="Prediabetic")
                        else:
                          return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="Prediabetic",gnb_pred_text="Prediabetic",pred_text="diabetic")
                    else: 
                        if (final_pred==0):
                           return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="Prediabetic",gnb_pred_text="diabetic",pred_text="Healthy")
                        elif (final_pred==1):
                           return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="Prediabetic",gnb_pred_text="diabetic",pred_text="Prediabetic")
                        else:
                           return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="Prediabetic",gnb_pred_text="diabetic",pred_text="diabetic")
                else:
                    if (gnb_pred == 0):
                        if (final_pred==0):
                          return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="diabetic",gnb_pred_text="Healthy",pred_text="Healthy")
                        elif (final_pred==1):
                          return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="diabetic",gnb_pred_text="Healthy",pred_text="Prediabetic")
                        else:
                          return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="diabetic",gnb_pred_text="Healthy",pred_text="diabetic")
                    elif(gnb_pred == 1):
                        if (final_pred==0):
                          return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="diabetic",gnb_pred_text="Prediabetic",pred_text="Healthy")
                        elif (final_pred==1):
                          return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="diabetic",gnb_pred_text="Prediabetic",pred_text="Prediabetic")
                        else:
                          return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="diabetic",gnb_pred_text="Prediabetic",pred_text="diabetic")
                    else: 
                        if (final_pred==0):
                          return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="diabetic",gnb_pred_text="diabetic",pred_text="Healthy")
                        elif (final_pred==1):
                          return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="diabetic",gnb_pred_text="diabetic",pred_text="Prediabetic")
                        else:
                          return render_template('mainPage.html', lr_pred_text="Prediabetic",knn_pred_text="diabetic",gnb_pred_text="diabetic",pred_text="diabetic")
        else:
            if (knn_pred == 0):
                    if (gnb_pred == 0):
                        if (final_pred==0):
                          return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="Healthy",gnb_pred_text="Healthy",pred_text="Healthy")
                        elif (final_pred==1):
                         return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="Healthy",gnb_pred_text="Healthy",pred_text="Prediabetic")
                        else:
                         return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="Healthy",gnb_pred_text="Healthy",pred_text="diabetic")
                    elif(gnb_pred == 1):
                        if (final_pred==0):
                            return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="Healthy",gnb_pred_text="Prediabetic",pred_text="Healthy")
                        elif (final_pred==1):
                            return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="Healthy",gnb_pred_text="Prediabetic",pred_text="Prediabetic")
                        else:
                            return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="Healthy",gnb_pred_text="Prediabetic",pred_text="diabetic")
                    else: 
                        if (final_pred==0):
                          return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="Healthy",gnb_pred_text="diabetic",pred_text="Healthy")
                        elif (final_pred==1):
                          return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="Healthy",gnb_pred_text="diabetic",pred_text="Prediabetic")
                        else:
                          return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="Healthy",gnb_pred_text="diabetic",pred_text="diabetic")
                  
            elif (knn_pred==1):
                    if (gnb_pred == 0):
                        if (final_pred==0):
                          return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="Prediabetic",gnb_pred_text="Healthy",pred_text="Healthy")
                        elif (final_pred==1):
                          return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="Prediabetic",gnb_pred_text="Healthy",pred_text="Prediabetic")
                        else:
                          return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="Prediabetic",gnb_pred_text="Healthy",pred_text="diabetic")
                    elif(gnb_pred == 1):
                        if (final_pred==0):
                          return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="Prediabetic",gnb_pred_text="Prediabetic",pred_text="Healthy")
                        elif (final_pred==1):
                          return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="Prediabetic",gnb_pred_text="Prediabetic",pred_text="Prediabetic")
                        else:
                          return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="Prediabetic",gnb_pred_text="Prediabetic",pred_text="diabetic")
                    else: 
                        if (final_pred==0):
                           return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="Prediabetic",gnb_pred_text="diabetic",pred_text="Healthy")
                        elif (final_pred==1):
                           return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="Prediabetic",gnb_pred_text="diabetic",pred_text="Prediabetic")
                        else:
                           return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="Prediabetic",gnb_pred_text="diabetic",pred_text="diabetic")
            else:
                    if (gnb_pred == 0):
                        if (final_pred==0):
                          return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="diabetic",gnb_pred_text="Healthy",pred_text="Healthy")
                        elif (final_pred==1):
                          return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="diabetic",gnb_pred_text="Healthy",pred_text="Prediabetic")
                        else:
                          return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="diabetic",gnb_pred_text="Healthy",pred_text="diabetic")
                    elif(gnb_pred == 1):
                        if (final_pred==0):
                          return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="diabetic",gnb_pred_text="Prediabetic",pred_text="Healthy")
                        elif (final_pred==1):
                          return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="diabetic",gnb_pred_text="Prediabetic",pred_text="Prediabetic")
                        else:
                          return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="diabetic",gnb_pred_text="Prediabetic",pred_text="diabetic")
                    else: 
                        if (final_pred==0):
                          return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="diabetic",gnb_pred_text="diabetic",pred_text="Healthy")
                        elif (final_pred==1):
                          return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="diabetic",gnb_pred_text="diabetic",pred_text="Prediabetic")
                        else:
                          return render_template('mainPage.html', lr_pred_text="diabetic",knn_pred_text="diabetic",gnb_pred_text="diabetic",pred_text="diabetic") 
                   
    #For rendering result on HTML file
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = lrmodel.predict(final_features)

    # output = round(prediction[0],2)

    # return render_template('mainPage.html',knn_pred_text='You are {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

    


    