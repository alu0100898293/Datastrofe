import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestCentroid, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist

def aplicar_agrupamiento(X, seed, parameters):
    """
    Función que ejecuta un pipeline con distintos tipos de algoritmos de regresion
    :param X: datos de variables predictoras
    :param seed: semilla para añadir variación en la división de los datos
    :param parameters: dict, diversos valores para los algoritmos de regresion
    :return:
    """
    
    # Agrupamiento basado en prototipos - K-means
    st.markdown("___")
    st.markdown("#### K-means")

    try:
        alg = KMeans(n_clusters=parameters['nClusters'], random_state=seed, init = "k-means++", max_iter = 500, n_init = 10)
        model = alg.fit(X)

        st.markdown("##### Centroides")
        centroids = pd.DataFrame(model.cluster_centers_, columns = X.columns)
        st.table(centroids)

        if parameters['wcss']:
            wcss = []
            for i in range(1, 10):
                kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 500, n_init = 10, random_state = seed)
                kmeans.fit(X)
                wcss.append(kmeans.inertia_)
            fig = go.Figure(data = go.Scatter(x = [1,2,3,4,5,6,7,8,9,10], y = wcss))
            st.markdown("##### WCSS vs. Número de clusters")
            fig.update_layout(
                            xaxis_title='Clusters',
                            yaxis_title='WCSS')
            st.plotly_chart(fig)

        if len(X.columns) == 2:
            st.markdown("##### Representación de clusters")
            X['Cluster'] = model.labels_
            fig = px.scatter(X, x=X.columns[0], y=X.columns[1], color='Cluster')
            st.plotly_chart(fig)
        
        elif len(X.columns) == 3:
            st.markdown("##### Representación de clusters")
            X['Cluster'] = model.labels_
            fig = px.scatter_3d(X, x=X.columns[0], y=X.columns[1], z=X.columns[2],
                color='Cluster', opacity = 0.8, size=X.columns[0], size_max=30)
            st.plotly_chart(fig)

    except Exception as e:
            st.error("Se produjo el siguiente error al crear el modelo:")
            st.error(e)
    
    ###############
    # Agrupamiento jerárquico
    st.markdown("___")
    st.markdown("#### Jeráquico")

    try:

        X_scaled = StandardScaler().fit_transform(X)
        
        dist_methods = []
        dist_methods.append("single")
        dist_methods.append("complete")
        dist_methods.append("average")
        dist_methods.append("ward")

        plt.title("Dendrograma")
        dist_matrix = linkage(X_scaled, method=parameters['dist'])
        dendro = dendrogram(dist_matrix)
        st.pyplot()


        c, coph_dists = cophenet(dist_matrix, pdist(X_scaled))
        st.markdown("##### Coeficiente de copheneteic")
        st.markdown("Distancia _" + parameters['dist']+ "_: " + str(round(c, 3)))
        
        if parameters['compDist']:
            st.markdown("##### Resto de coeficientes según método de distancia")
            dist_methods.remove(parameters['dist'])
            for d_method in dist_methods:
                dist_matrix = linkage(X_scaled, method=d_method)
                c, coph_dists = cophenet(dist_matrix, pdist(X_scaled))
                st.markdown("Distancia _" + d_method + "_: " + str(round(c, 3)))
        


        alg = AgglomerativeClustering(n_clusters=parameters['nClusters_h'], affinity='euclidean', linkage=parameters['dist'])
        model = alg.fit_predict(X_scaled)
        st.markdown("##### Centroides")
        clf = NearestCentroid()
        clf.fit(X_scaled, model)
        st.table(centroids)

    except Exception as e:
        st.error("Se produjo el siguiente error al crear el modelo:")    
        st.error(e)
        
    ###############
    # Agrupamiento basado en desidad
    st.markdown("___")
    st.markdown("#### DBScan")

    try:
            
        alg = DBSCAN(eps=parameters['eps'], min_samples=parameters['minPts'])
        model = alg.fit(X)
        labels = model.labels_

        st.markdown("##### Resumen")
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        cluster_counter = Counter(labels)

        st.markdown("Número de clusters estimados: %d" % n_clusters_)
        st.markdown("Puntos estimados como ruido: %d" % n_noise_)
        if len(cluster_counter) > 1:
            st.markdown("Coeficiente de silueta: %0.3f" % metrics.silhouette_score(X, labels))
        st.markdown("Clusters:")

        for key in cluster_counter.keys():
            st.markdown("- **Cluster " + str(key) + "**: " +  str(cluster_counter[key]) + " elementos")

        if len(X.columns) == 3:
            st.markdown("##### Representación de clusters")
            X['Cluster'] = model.labels_
            fig = px.scatter(X, x=X.columns[0], y=X.columns[1], color='Cluster')
            st.plotly_chart(fig)
        
        elif len(X.columns) == 4:
            st.markdown("##### Representación de clusters")
            X['Cluster'] = model.labels_
            fig = px.scatter_3d(X, x=X.columns[0], y=X.columns[1], z=X.columns[2],
                color='Cluster', opacity = 0.8, size=X.columns[0], size_max=30)
            st.plotly_chart(fig)

        if parameters['knnDist']:
            st.markdown("##### Distancia entre " + str(parameters['minPts']) +" vecinos más cercanos")
            neigh = NearestNeighbors(n_neighbors=parameters['minPts'])
            nbrs = neigh.fit(X)
            distances, indices = nbrs.kneighbors(X)
            distances = np.sort(distances, axis=0)
            distances = distances[:,1]
            plt.plot(distances)
            st.pyplot()
        
    except Exception as e:
            st.error("Se produjo el siguiente error al crear el modelo:")
            st.error(e)


   