import streamlit as st
import plotly.express as px
from src.exportar_imagen import mostrar_formato_exportacion


def control_graficos(chart_type, df, dropdown_options, template):
    """
    Función que muestra los diferentes parámetros aceptados según el tipo de gráfico elegido
    :param chart_type: str, nombre del gráfico
    :param df: set de datos importado
    :param dropdown_options: lista con los nombres de las columnas
    :param template: str, representación del tema elegido
    :return:
    """
    length_of_options = len(dropdown_options)
    length_of_options -= 1

    plot = px.scatter()

    if chart_type == 'Grafico de dispersion':
        st.sidebar.subheader("Gráfico de dispersión")

        try:
            x_values = st.sidebar.selectbox('Eje X', index=length_of_options,options=dropdown_options)
            y_values = st.sidebar.selectbox('Eje Y',index=length_of_options, options=dropdown_options)
            color_value = st.sidebar.selectbox("Color", index=length_of_options,options=dropdown_options)
            symbol_value = st.sidebar.selectbox("Símbolo",index=length_of_options, options=dropdown_options)
            marginaly = st.sidebar.selectbox("Gráfico marginal", index=0,options=[ None, 'box',
                                                                         'violin', 'histogram'])
            log_x = st.sidebar.selectbox('Escala log. en x', options=[False, True])
            log_y = st.sidebar.selectbox('Escala log. en y', options=[False, True])
            title = st.sidebar.text_input(label='Título de gráfico')
            plot = px.scatter(data_frame=df,
                              x=x_values,
                              y=y_values,
                              color=color_value,
                              symbol=symbol_value,
                              marginal_y=marginaly, log_x=log_x, log_y=log_y,
                              template=template, title=title)

        except Exception as e:
            print(e)

    if chart_type == 'Histograma':
        st.sidebar.subheader("Histograma")

        try:
            x_values = st.sidebar.selectbox('Eje X', index=length_of_options,options=dropdown_options)
            y_values = st.sidebar.selectbox('Eje y',index=length_of_options, options=dropdown_options)
            nbins = st.sidebar.number_input(label='Número de contenedores', min_value=2, value=5)
            color_value = st.sidebar.selectbox("Color", index=length_of_options,options=dropdown_options)

            hist_func = st.sidebar.selectbox('Función de agregación del histograma', index=0,
                                             options=['count','sum', 'avg', 'min', 'max'])
            title = st.sidebar.text_input(label='Título del gráfico')
            plot = px.histogram(data_frame=df,
                                x=x_values, y=y_values,
                                nbins=nbins,
                                color=color_value,
                                histfunc=hist_func,
                                template=template, title=title)

        except Exception as e:
            print(e)

    if chart_type == 'Grafico por sectores':
        st.sidebar.subheader('Gráfico por sectores')

        try:
            name_value = st.sidebar.selectbox(label='Nombre (La columna alegida debería ser categórica)', index=length_of_options, options=dropdown_options)
            color_value = st.sidebar.selectbox(label='Color (La columna alegida debería ser categórica)', index=length_of_options, options=dropdown_options)
            value = st.sidebar.selectbox("Valor", index=length_of_options, options=dropdown_options)
            hole = st.sidebar.selectbox('Nombre en eje y', options=[True, False])
            title = st.sidebar.text_input(label='Título del gráfico')

            plot = px.pie(data_frame=df,names=name_value,hole=hole,
                          values=value,color=color_value, title=title)

        except Exception as e:
            print(e)

    if chart_type == 'Grafico de lineas':
        st.sidebar.subheader("Gráfico de lineas")

        try:
            x_values = st.sidebar.selectbox('Eje X', index=length_of_options, options=dropdown_options)
            y_values = st.sidebar.selectbox('Eje Y', options=dropdown_options)
            color_value = st.sidebar.selectbox("Color", index=length_of_options, options=dropdown_options)
            #line_group = st.sidebar.selectbox("Line group", options=dropdown_options)
            #line_dash = st.sidebar.selectbox("Line dash", index=length_of_options,options=dropdown_options)
            log_x = st.sidebar.selectbox('Escala log. en x', options=[False, True])
            log_y = st.sidebar.selectbox('Escala log. en y', options=[False, True])
            title = st.sidebar.text_input(label='Título del gráfico')
            plot = px.line(data_frame=df,
                           #line_group=line_group,
                           #line_dash=line_dash,
                           x=x_values,y=y_values,
                           color=color_value,
                           log_x=log_x,
                           log_y=log_y,
                           template=template,
                           title=title)
        except Exception as e:
            print(e)

    if chart_type == 'Grafico de barras':
        st.sidebar.subheader('Gráfico de barras')

        try:
            x_values = st.sidebar.selectbox('Eje X', index=length_of_options, options=dropdown_options)
            y_values = st.sidebar.selectbox('Ejec Y', index=length_of_options, options=dropdown_options)
            color_value = st.sidebar.selectbox("Color", index=length_of_options, options=dropdown_options)
            hover_name_value = st.sidebar.selectbox("Nombre superpuesto", index=length_of_options, options=dropdown_options)
            barmode = st.sidebar.selectbox('Modo de barra', options=['stack', 'group', 'overlay','relative'], index=3)
            title = st.sidebar.text_input(label='Título del gráfico')

            plot = px.bar(data_frame=df, 
                            x=x_values, y=y_values, 
                            color=color_value, 
                            template=template,
                            hover_name=hover_name_value, 
                            barmode=barmode,
                            title=title)

        except Exception as e:
            print(e)
    
    if chart_type == 'Grafico de violin':
        st.sidebar.subheader('Gráfico de Violín')
    
        try:
            x_values = st.sidebar.selectbox('Eje x', index=length_of_options,options=dropdown_options)
            y_values = st.sidebar.selectbox('Eje y',index=length_of_options, options=dropdown_options)
            color_value = st.sidebar.selectbox("Color", index=length_of_options,options=dropdown_options)
            violinmode = st.sidebar.selectbox('Modo de violín', options=['group', 'overlay'])
            box = st.sidebar.selectbox("Mostrar caja", options=[False, True])
            outliers = st.sidebar.selectbox('Mostrar puntos', options=[False, 'all', 'outliers', 'suspectedoutliers'])
            log_x = st.sidebar.selectbox('Escala log. en x', options=[False, True])
            log_y = st.sidebar.selectbox('Escala log. en y', options=[False, True])
            title = st.sidebar.text_input(label='Título del gráfico')
            plot = px.violin(data_frame=df,x=x_values,
                             y=y_values,color=color_value,
                             box=box,
                             log_x=log_x, log_y=log_y,
                             violinmode=violinmode,points=outliers,
                             template=template, title=title)
    
        except Exception as e:
            print(e)
    
    if chart_type == 'Grafico de cajas':
        st.sidebar.subheader('Gráfico de cajas')
    
        try:
            x_values = st.sidebar.selectbox('Eje x', index=length_of_options, options=dropdown_options)
            y_values = st.sidebar.selectbox('Eje y', index=length_of_options, options=dropdown_options)
            color_value = st.sidebar.selectbox("Color", index=length_of_options, options=dropdown_options)
            boxmode = st.sidebar.selectbox('Modo de caja', options=['group', 'overlay'])
            outliers = st.sidebar.selectbox('Mostrar puntos', options=[False, 'all', 'outliers', 'suspectedoutliers'])
            log_x = st.sidebar.selectbox('Escala log. en x', options=[False, True])
            log_y = st.sidebar.selectbox('Escala log. en y', options=[False, True])
            notched = st.sidebar.selectbox('Mostrar muescas', options=[False, True])
            title = st.sidebar.text_input(label='Título del gráfico')
            plot = px.box(data_frame=df, x=x_values,
                          y=y_values, color=color_value,
                          notched=notched,
                          log_x=log_x, log_y=log_y, boxmode=boxmode, points=outliers,
                          template=template, title=title)
    
        except Exception as e:
            print(e)
    
    #if chart_type == 'Sunburst':
    #    st.sidebar.subheader('Sunburst Settings')
    #
    #    try:
    #        path_value = st.sidebar.multiselect(label='Path', options=dropdown_options)
    #        color_value = st.sidebar.selectbox(label='Color', options=dropdown_options)
    #        value = st.sidebar.selectbox("Value", index=length_of_options, options=dropdown_options)
    #        title = st.sidebar.text_input(label='Title of chart')
    #
    #        plot = px.sunburst(data_frame=df,path=path_value,values=value,
    #                           color=color_value, title=title )
    #
    #    except Exception as e:
    #        print(e)
    #
    #if chart_type == 'Tree maps':
    #    st.sidebar.subheader('Tree maps Settings')
    #
    #    try:
    #        path_value = st.sidebar.multiselect(label='Path', options=dropdown_options)
    #        color_value = st.sidebar.selectbox(label='Color', options=dropdown_options)
    #        value = st.sidebar.selectbox("Value", index=length_of_options, options=dropdown_options)
    #        title = st.sidebar.text_input(label='Title of chart')
    #
    #        plot = px.treemap(data_frame=df,path=path_value,values=value,
    #                          color=color_value, title=title )
    #
    #    except Exception as e:
    #        print(e)
    #
    #
    #
    #if chart_type == 'Density contour':
    #    st.sidebar.subheader("Density contour Settings")
    #
    #    try:
    #        x_values = st.sidebar.selectbox('X axis', index=length_of_options,options=dropdown_options)
    #        y_values = st.sidebar.selectbox('Y axis',index=length_of_options, options=dropdown_options)
    #        z_value = st.sidebar.selectbox("Z axis", index=length_of_options, options=dropdown_options)
    #        color_value = st.sidebar.selectbox("Color", index=length_of_options,options=dropdown_options)
    #        hist_func = st.sidebar.selectbox('Histogram aggregation function', index=0,
    #                                         options=['count', 'sum', 'avg', 'min', 'max'])
    #        histnorm = st.sidebar.selectbox('Hist norm', options=[None, 'percent', 'probability', 'density',
    #                                                              'probability density'], index=0)
    #        hover_name_value = st.sidebar.selectbox("Hover name", index=length_of_options,options=dropdown_options)
    #        facet_row_value = st.sidebar.selectbox("Facet row",index=length_of_options, options=dropdown_options,)
    #        facet_column_value = st.sidebar.selectbox("Facet column", index=length_of_options,
    #                                                  options=dropdown_options)
    #        marginalx = st.sidebar.selectbox("Marginal X", index=2,options=['rug', 'box', None,
    #                                                                     'violin', 'histogram'])
    #        marginaly = st.sidebar.selectbox("Marginal Y", index=2,options=['rug', 'box', None,
    #                                                                     'violin', 'histogram'])
    #        log_x = st.sidebar.selectbox('Log axis on x', options=[True, False],index=1)
    #        log_y = st.sidebar.selectbox('Log axis on y', options=[True, False], index=1)
    #        title = st.sidebar.text_input(label='Title of chart')
    #        plot = px.density_contour(data_frame=df,x=x_values,y=y_values, color=color_value,
    #                                  z=z_value, histfunc=hist_func,histnorm=histnorm,
    #                                  hover_name=hover_name_value,facet_row=facet_row_value,
    #                                  facet_col=facet_column_value,log_x=log_x,
    #                                  log_y=log_y,marginal_y=marginaly, marginal_x=marginalx,
    #                                  template=template, title=title)
    #
    #    except Exception as e:
    #        print(e)
    #
    #if chart_type == 'Density heatmaps':
    #    st.sidebar.subheader("Density heatmap Settings")
    #
    #    try:
    #        x_values = st.sidebar.selectbox('X axis', index=length_of_options, options=dropdown_options)
    #        y_values = st.sidebar.selectbox('Y axis', index=length_of_options, options=dropdown_options)
    #        z_value = st.sidebar.selectbox("Z axis", index=length_of_options, options=dropdown_options)
    #        hist_func = st.sidebar.selectbox('Histogram aggregation function', index=0,
    #                                         options=['count', 'sum', 'avg', 'min', 'max'])
    #        histnorm = st.sidebar.selectbox('Hist norm', options=[None, 'percent', 'probability', 'density',
    #                                                              'probability density'], index=0)
    #        hover_name_value = st.sidebar.selectbox("Hover name", index=length_of_options, options=dropdown_options)
    #        facet_row_value = st.sidebar.selectbox("Facet row", index=length_of_options, options=dropdown_options, )
    #        facet_column_value = st.sidebar.selectbox("Facet column", index=length_of_options,
    #                                                  options=dropdown_options)
    #        marginalx = st.sidebar.selectbox("Marginal X", index=2, options=['rug', 'box', None,
    #                                                                         'violin', 'histogram'])
    #        marginaly = st.sidebar.selectbox("Marginal Y", index=2, options=['rug', 'box', None,
    #                                                                         'violin', 'histogram'])
    #        log_x = st.sidebar.selectbox('Log axis on x', options=[True, False], index=1)
    #        log_y = st.sidebar.selectbox('Log axis on y', options=[True, False], index=1)
    #        title = st.sidebar.text_input(label='Title of chart')
    #        plot = px.density_heatmap(data_frame=df, x=x_values, y=y_values,
    #                                  z=z_value, histfunc=hist_func, histnorm=histnorm,
    #                                  hover_name=hover_name_value, facet_row=facet_row_value,
    #                                  facet_col=facet_column_value, log_x=log_x,
    #                                  log_y=log_y, marginal_y=marginaly, marginal_x=marginalx,
    #                                  template=template, title=title)
    #
    #    except Exception as e:
    #        print(e)

    st.subheader("Gráfico")
    st.plotly_chart(plot)
    mostrar_formato_exportacion(plot)

