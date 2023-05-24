import streamlit as st
import pysolr

# Create a connection to Solr
from utils import OhdsiManager, SolrManager, IdManager
myOhdsi = OhdsiManager(database = "ohdsi_cumc_2022q4r1")
mySolr = SolrManager()
myId = IdManager(type = "epic", database = "ohdsi_cumc_2022q4r1") # make sure the datebase is the same as myOhdsi one.


def get_cohort(epic_id):
        myId.addIdList([epic_id])
        myId.getAllIds()
        person_id = myId.IdMappingDf.values[0][0]
        empi = myId.IdMappingDf.values[0][1].strip()
        cohort_df = myId.IdMappingDf
        return cohort_df

def get_demo_df(epic_id, demo_source):
    cohort_df = get_cohort(epic_id)
    df = myOhdsi.get_demo(cohort_df, source=demo_source)
    return df

def get_note_df(epic_id, note_source, title, provider_name):
    df = mySolr.get_note(empi=epic_id, source=note_source, title=title, provider_name=provider_name)
    return df 


tab_name_list = ['demo', 'note']
tab_list = st.tabs(tab_name_list)

with st.sidebar:
    st.title('Columbia Patient View')
    st.write('In-house application based on Solr and OHDSI')

    epic_id = st.text_input('Enter Epic ID: ')

    config_option = st.selectbox("config options for...", ["Demo", "Note"])

    title = None
    provider_name = None
    demo_source = False
    note_source = False
    # Checkbox inputs
    if config_option == 'Demo':
        # Conditional display based on checkbox selection
        demo_source = st.checkbox("display source table for demo")
    if config_option == 'Note':
        note_source = st.checkbox("display source table for note")
        title = st.text_input('titles seperated by ";"')
        title = title.split(';')
        provider_name = st.text_input('providers seperated by ";"')
        provider_name = provider_name.split(';')


    if st.button("View") and epic_id!='':
        demo_df = get_demo_df(epic_id, demo_source)
        note_df = get_note_df(epic_id=epic_id, note_source=note_source, title=title, provider_name=provider_name)
        



with tab_list[0]:
    try:
        st.write(demo_df)
    except:
        st.write('demo_df is not available')

with tab_list[1]:
    try:
        st.write(note_df)
    except:
        st.write('note_df is not available')
     
