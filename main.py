import streamlit as st
st.set_page_config(page_title="Graph Showcase", page_icon="📊", layout="wide")

from streamlit_option_menu import option_menu
from Page_Files import Contact_Us, Home, Model, Visualization, Evaluation, Comparison


class MultiApp:
    def __init__(self):
        self.apps()
    def add_app(self,title,function):
        self.apps.append({
            "title":title,
            "function":function
        })
        
    def run():
        with st.sidebar:
            selected=option_menu(
                menu_title="Sentiment Analyser",
                options=["Home", "Model", "Visualization","Evaluation","Comparison","Contact_Us"],
                icons=["house","book", "graph-up","file-earmark-break-fill","clipboard2-data-fill","chat-text-fill"],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container":{"padding":"5,!important","background-color":"black"},
                    "icon":{"color":"white","font-size":"23px"},
                    "nav-link":{"color":"white","font-size":"20px","text-align":"left"},
                    "nav-link-selected":{"background-color":"maroon"}
                    }
            )
        if selected=="Home":
            Home.run()
        if selected=="Model":
            Model.run()
        if selected=="Visualization":
            Visualization.run()
        if selected=="Evaluation":
            Evaluation.run()
        if selected=="Comparison":
            Comparison.run()
        if selected=="Contact_Us":
            Contact_Us.run()
        
    run()