#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def knn(target_variable,train,test,neighbors,p,weight):

    ## establish target variable    
    target_variable = str(target_variable)
    
    ## develop a for loop if we want to include weights for the model
    
    if weight == True:
    
        list_index = []
        l2 = []
        label = []
        for n in range(len(test)):
            for i in range(len(train)):
                list_index.append(test.reset_index().iloc[n]['index'])
                l = test.drop(target_variable, axis = 1).iloc[n] - train.drop(target_variable, axis = 1).iloc[i] ## distance of the points
                distance = (sum(abs(l)**p))**(1/p)
                if distance == 0:
                    l2.append(0)
                else:
                    l2.append(1/distance) ## formula for the distance depending on which type of distance
                label.append(train.iloc[i]['class'])
        data = pd.DataFrame(list(zip(list_index , label , l2)), columns = {'index_key' , 'l2_distance' , 'label'})


        list_index = []
        #l2 = []
        label = []
        for n in data.index_key.unique():
            d1 = data[data['index_key'] == n].sort_values(by = 'l2_distance', ascending = False).head(neighbors)
            label.append(d1.groupby('label').sum().sort_values(by = 'l2_distance', ascending = False).head(1).index[0])
            list_index.append(d1['index_key'].iloc[0])
        data = pd.DataFrame(list(zip(list_index , label)), columns = {'index_key', 'label'})
        data = test.reset_index().merge(data, left_on = 'index' , right_on = 'index_key').drop(['index' , 'index_key'], axis = 1).rename(columns = {'label'  : 'predicted_label'})

        for n in data.index:
            if data.loc[n , 'class'] == data.loc[n , 'predicted_label']:
                data.loc[n, 'predicted_label_error'] = 0
            else:
                data.loc[n , 'predicted_label_error'] = 1

        error_rate = data['predicted_label_error'].sum()/len(data)

        return data, error_rate

    else:
        list_index = []
        l2 = []
        label = []
        for n in range(len(test)):
            for i in range(len(train)):
                list_index.append(test.reset_index().iloc[n]['index'])
                l = test.drop(target_variable, axis = 1).iloc[n] - train.drop(target_variable, axis = 1).iloc[i] ## distance of the points
                distance = (sum(abs(l)**p))**(1/p)
                l2.append(distance)
                label.append(train.iloc[i]['class'])
        data = pd.DataFrame(list(zip(list_index , label , l2)), columns = {'index_key' , 'l2_distance' , 'label'})


        list_index = []
        #l2 = []
        label = []
        for n in data.index_key.unique():
            d1 = data[data['index_key'] == n].sort_values(by = 'l2_distance').head(neighbors)
            label.append(d1['label'].value_counts().index[0])
            list_index.append(d1['index_key'].iloc[0])
        data = pd.DataFrame(list(zip(list_index , label)), columns = {'index_key', 'label'})
        data = test.reset_index().merge(data, left_on = 'index' , right_on = 'index_key').drop(['index' , 'index_key'], axis = 1).rename(columns = {'label'  : 'predicted_label'})

# get error count
        for n in data.index:
            if data.loc[n , 'class'] == data.loc[n , 'predicted_label']:
                data.loc[n, 'predicted_label_error'] = 0
            else:
                data.loc[n , 'predicted_label_error'] = 1

        error_rate = data['predicted_label_error'].sum()/len(data)


        return data, error_rate

