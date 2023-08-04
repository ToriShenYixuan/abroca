import os
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox
from shapely.geometry import Polygon 
import time
from shapely.geometry import LineString 
import shapely

#use this one
def resample(df,demographic, bin1, bin2):
    user_list = pd.DataFrame({'studentid': df['studentid'].unique()})['studentid']
    split=(df[df[demographic]==bin1]["studentid"]).nunique()
    selected_id = np.random.choice(user_list, split, replace=False)
    df=df.copy()
    df.loc[:,demographic] = np.where(df['studentid'].isin(selected_id), bin1, bin2)
    
    return df


def calculate_roc(df,dem,demographic,actual,predicted, bin2):
    if dem != "all":
        df=df[df[demographic]==dem]

    test_data_true = np.array(df[actual])
    test_recon = np.array(df[predicted])
    fpr, tpr, threshold = roc_curve(test_data_true, test_recon)

    if dem == bin2:    
        indices=np.argsort(fpr)[::-1]
        fpr=fpr[indices]
        tpr=tpr[indices]

    auc_score=0
    auc_score=round(roc_auc_score(test_data_true, test_recon),4)
    return fpr, tpr, auc_score


def greaterthan(point_a,point_b):
    if point_a[0]>=point_b[0] and point_a[1] >=point_b[1]:
        return True
    else:
        return False


def binary_split(arr,x,hightolow=False):
    low = 0
    high = len(arr) - 1
    mid = 0
    
    while low <= high:
        mid = (high + low) // 2
        criteria=greaterthan(x,arr[mid])
        if hightolow:
            criteria=not criteria
        
        if criteria:
            low = mid + 1

        elif not criteria:
            
            high = mid - 1
    mid=mid+criteria
    returnlist=[arr[0:mid]+[x],[x]+arr[mid:len(arr)]]
    #print(f"splitting at {mid}, into {returnlist[hightolow],returnlist[not hightolow]}")
    return returnlist[hightolow],returnlist[not hightolow]



def get_abroca_area(first, second):
    first_line=LineString(first)
    second_line=LineString(second)
    intersect_multi=first_line.intersection(second_line)
    points=intersect_multi.geoms
    intersects=[(p.coords[0]) for p in points]
    intersects.sort(reverse=True)
    first_points=first
    second_points=second
    poly=[]
    for i in range(1,len(intersects)):
        intersecting_point=intersects[i]
        # print(f"intersecting point=({intersects[i]})")
        first_points, poly_f = binary_split(first_points,intersecting_point)
        second_points, poly_s = binary_split(second_points,intersecting_point, hightolow=True)
        poly.append(Polygon(poly_f+poly_s))
        if i==len(intersects)-2:
            poly.append(Polygon(first_points+second_points))
    area=0
    for i in poly:
        area+=i.area
    return area


    
def get_bootstraps(df, demographic, bin1, bin2, bootstrap, actual, predicted):
    print("currently on:")
    abroca_boot=[]
    if bin2!="other":
        df=df[df[demographic].isin([bin1,bin2])]
    for i in range(bootstrap):
        if i%50==0:
            print(f"bootstrap: {i+1}")
        df_boot=resample(df, demographic, bin1, bin2)
        polygon_points=[]
        for dem in [bin1,bin2]:
            fpr, tpr, auc_score= calculate_roc(df_boot, dem, demographic, actual, predicted, bin2)
            polygon_points.append(list(zip(fpr,tpr)))
        area=get_abroca_area(polygon_points[0],polygon_points[1])
        abroca_boot.append(area)

    return abroca_boot


def get_graph(df, demographic, bin1, bin2, actual, predicted, bootstrap, abroca_boot, getGraph=True):
    polygon_points=[]
    poly_x=[]
    poly_y=[]
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"{df.name} Abroca Output")
    auc=[]
    # splitting the df by gender and adding the sub dfs to the dictionary

    for dem in [bin1,bin2]:

        fpr, tpr, auc_score = calculate_roc(df, dem, demographic, actual, predicted, bin2)

        auc.append(auc_score)
        poly_x.extend(list(fpr))
        poly_y.extend(list(tpr))
        polygon_points.append(list(zip(fpr,tpr)))
        
        y_text=0.15-(dem == bin2)*0.05
        axs[0].plot(fpr, tpr, label = f"{demographic}={str(dem)}")
        axs[0].text(0.2,y_text,f"auc for {demographic}={str(dem)}: {str(auc_score)}")
    
    area=get_abroca_area(polygon_points[0],polygon_points[1])

    if getGraph: 
        path=os.path.join("/Users/shent/Desktop/summer23/fairness/abroca_boot/output/"+df.name)
        if not os.path.exists(path):

            # Create a new directory because it does not exist
            os.makedirs(path)
        axs[0].fill(poly_x,poly_y,"y",)                                              # filling the area between curves
        axs[0].legend(loc="upper left")                                                              # adding legend for labels
        axs[0].set_title(f"ROC curves for test dataset on {demographic}")
        axs[0].text(0.33,0,"area between curves (ABROCA): "+str(round(area,6)))      # adding display text for calculated area
        axs[0].set_xlabel("false positive rate")
        axs[0].set_ylabel("true positive rate")

        if abroca_boot:
            axs[1].hist(abroca_boot, 50, density=False)
            per_nn=round(np.percentile(abroca_boot,99),4)
            x_max=axs[1].get_xlim()[1]
            y_max=axs[1].get_ylim()[1]
            axs[1].vlines(x=per_nn, ymin=0,ymax=y_max*0.8,color='green', linestyle='--') 
            axs[1].text(per_nn+0.01*x_max, y_max*0.6, f'99th = {per_nn}', color='green', rotation=90)
            axs[1].set_title(f"{bootstrap}-Bootstrap Distribution ABROCA statistics")
            axs[1].set_xlabel("ABROCA statistic")
            axs[1].set_ylabel("counts")

            if area<x_max:
                axs[1].vlines(x=area, ymin=0,ymax=y_max*0.8, color='orange', linestyle='--') 
                axs[1].text(area+0.01*x_max, y_max*0.525, f'observed = {round(area,4)}', color='orange', rotation=90)
                p=len([x for x in abroca_boot if x > area])/len(abroca_boot)
                p=round(p,6) if p< 0.5 else round(1-p,6)
            else:
                axs[1].text(x_max*0.6,y_max*0.87,f"*Observed value={round(area,4)}, \n outside bootstrap distribution", fontsize=8)
                p=0  
            
            p_string=f"p-value={p}*" if p<=0.05 else f"p-value={p}"
            axs[1].text(x_max*0.7,y_max*0.95,p_string)
            image_name=os.path.join(path+f"/{demographic}_{bootstrap}_{bin1}_{bin2}_boots_abroca.png")
            fig.savefig(image_name)            
        else:
            axs[1].remove()
            image_name=os.path.join(path+f"/{demographic}_{bin1}_{bin2}_abroca.png")
            fig.savefig(image_name)
        
        print(f"image saved as {image_name}")
    plt.close()
    return area
        
def processdf(df,demographic,bin1):
    df.loc[df[demographic]!=bin1,demographic]="other"
    return df


def ABROCA(df, demographic, actual, predicted, bin1, bin2, bootstrap=False, getGraph=True):
    name=df.name
    df=df.copy()
    df.name=name
    df.dropna(subset=[demographic], inplace=True)
    df[demographic]=df[demographic].astype(int).astype(str)
    bin1=str(bin1)
    if bin2=="other":
        df=processdf(df,demographic,bin1)

    bin2=str(bin2)

    abroca_boot=[]
    if bootstrap:
        starttime=time.time()
        abroca_boot = get_bootstraps(df, demographic, bin1, bin2, bootstrap, actual, predicted)
        totaltime=time.time()-starttime
        print(f"bootstrap took a total of {totaltime} seconds")
    area=get_graph(df, demographic, bin1, bin2, actual, predicted, bootstrap, abroca_boot, getGraph)
    # totaltime=time.time()-starttime-totaltime
    # print(f"generating graph took a total of {totaltime} seconds")
    return area


