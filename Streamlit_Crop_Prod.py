# Add import statements for new libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_data():
    try:
        data = pd.read_csv('Crop_Production_Statistics.csv')
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def main():
    st.title('Crop Production Statistics Analysis')
    st.sidebar.title('Filters')

    # Load the dataset
    data = load_data()

    if data is not None:
        # Display the dataset
        st.write('### Dataset')
        st.write(data)

        # Filter by state
        selected_state = st.sidebar.selectbox('Select State', data['State'].unique())
        filtered_data = data[data['State'] == selected_state]

        # Filter by crop year
        selected_crop_year = st.sidebar.slider('Select Crop Year', min_value=data['Crop_Year'].min(), max_value=data['Crop_Year'].max(), value=(data['Crop_Year'].min(), data['Crop_Year'].max()))
        filtered_data = filtered_data[(filtered_data['Crop_Year'] >= selected_crop_year[0]) & (filtered_data['Crop_Year'] <= selected_crop_year[1])]

        # Filter by crop
        selected_crop = st.sidebar.selectbox('Select Crop', filtered_data['Crop'].unique())
        filtered_data = filtered_data[filtered_data['Crop'] == selected_crop]

        # Filter by season
        selected_season = st.sidebar.selectbox('Select Season', filtered_data['Season'].unique())
        filtered_data = filtered_data[filtered_data['Season'] == selected_season]

        # Display filtered dataset
        st.write('### Filtered Dataset')
        st.write(filtered_data)

        # Visualizations
        st.write('### Data Visualization')

        # Distribution of Production
        st.write('#### Distribution of Production')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(filtered_data['Production'], bins=20, kde=True, ax=ax)
        plt.xlabel('Production')
        plt.ylabel('Count')
        plt.title('Distribution of Production')
        st.pyplot(fig)

        # Distribution of Yield
        st.write('#### Distribution of Yield')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(filtered_data['Yield'], bins=20, kde=True, ax=ax)
        plt.xlabel('Yield')
        plt.ylabel('Count')
        plt.title('Distribution of Yield')
        st.pyplot(fig)

        # Scatter plot of Production vs Yield
        st.write('#### Production vs Yield')
        fig = px.scatter(filtered_data, x='Production', y='Yield', hover_name='Crop', color='Season')
        st.plotly_chart(fig)

        # Summary statistics
        st.write('### Summary Statistics')
        st.write(filtered_data[['Production', 'Yield']].describe())

        # Data Export
        st.write('### Data Export')
        export_format = st.radio('Select export format:', ('CSV', 'Excel'))
        if st.button('Export Filtered Data'):
            if export_format == 'CSV':
                csv_data = filtered_data.to_csv(index=False)
                st.download_button(label='Download CSV', data=csv_data, file_name='filtered_data.csv', mime='text/csv')
            elif export_format == 'Excel':
                excel_data = filtered_data.to_excel(index=False)
                st.download_button(label='Download Excel', data=excel_data, file_name='filtered_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        # Machine Learning Model
        st.write('### Machine Learning Model')

        # Select features and target variable
        features = ['Crop_Year', 'Area ']  # Add relevant features
        target = 'Production'

        # Prepare data
        X = filtered_data[features]
        y = filtered_data[target]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate the model
        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
        test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

        st.write(f'Training RMSE: {train_rmse:.2f}')
        st.write(f'Testing RMSE: {test_rmse:.2f}')

        # Make predictions
        prediction_input = st.sidebar.text_input('Enter input for prediction (e.g., Crop_Year, Area)')
        if prediction_input:
            prediction_input = list(map(float, prediction_input.split(',')))  # Convert input to list of floats
            prediction = model.predict([prediction_input])[0]
            st.write(f'Predicted Production: {prediction:.2f}')

    else:
        st.error("Failed to load dataset.")

if __name__ == '__main__':
    main()
