
import pandas as pd
import configparser
import pyodbc
import sqlalchemy
import urllib
import uuid
import pysolr
import sys
import time


class SqlConnector():
    def __init__(self,configFile='/projects/phi/cl3720/db.conf',database='ohdsi_cumc_2022q4r1'):
        self.config = configparser.ConfigParser()
        self.config.read(configFile)
        # self.config.sections()
        self.server = self.config['ELILEX']['server']
        self.database = database
        self.username = self.config['ELILEX']['username']
        self.password = self.config['ELILEX']['password']
        self.cnxn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};SERVER='+self.server+';DATABASE='+self.database+';UID='+self.username+';PWD='+ self.password)
        params = 'Driver={ODBC Driver 17 for SQL Server};SERVER='+self.server+';DATABASE='+self.database+';UID='+self.username+';PWD='+ self.password
        db_params = urllib.parse.quote_plus(params)
        self.engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect={}".format(db_params))
    def getCnxn(self):
        cursor = self.cnxn.cursor()
        return self.cnxn, cursor
    
    def getEngine(self):
        # cursor = cnxn.cursor()
        return self.engine
    
class OhdsiManager():
    def __init__(self,configFile='/projects/phi/cl3720/db.conf',database='ohdsi_cumc_2021q3r1'):
        self.sqlconnector = SqlConnector(configFile,database)
        self.engine = self.sqlconnector.getEngine()
        self.cnxn, self.cursor = self.sqlconnector.getCnxn()

    def get_cohort(self, top_x = 10, ancestor_concept_id = "441840", not_in = False, year_of_birth_start = 1980, year_of_birth_end = 2000, num_visits = 10, observation_period_days = 365 * 5):
        '''
        Clinical Finding : 441840
        Genetic disease : 138875
        '''
        if not_in:
            sql_in = 'NOT IN'
        else:
            sql_in = 'IN'
        sql = '''
        SELECT  TOP {top_x}
                p.person_id
                FROM person p
                JOIN visit_occurrence v
                ON p.person_id = v.person_id
                JOIN condition_occurrence co
                on p.person_id = co.person_id AND v.visit_occurrence_id = co.visit_occurrence_id
                JOIN concept_ancestor ca
                on ca.descendant_concept_id = co.condition_concept_id
                WHERE 
                v.visit_start_date < DATEADD(year, 18, p.birth_datetime) AND v.visit_start_date >= DATEADD(year, 15, p.birth_datetime) AND
                v.visit_concept_id IN (9202) AND -- only consider outpatient visits
                p.year_of_birth >= {year_of_birth_start} AND p.year_of_birth < {year_of_birth_end} AND
                ca.ancestor_concept_id {sql_in} ({ancestor_concept_id}) -- Genetic disease
                GROUP BY 
                p.person_id, p.birth_datetime
                HAVING 
                COUNT(DISTINCT v.visit_occurrence_id) >= {num_visits}
                AND DATEDIFF(DAY, MIN(v.visit_start_date), MAX(v.visit_start_date))+1 >= {observation_period_days}
        '''.format(top_x = top_x, sql_in=sql_in, year_of_birth_start=year_of_birth_start, year_of_birth_end=year_of_birth_end,num_visits=num_visits,observation_period_days=observation_period_days,ancestor_concept_id=ancestor_concept_id)
        cohort_df = pd.read_sql(sql,self.cnxn)
        return cohort_df

    def get_demo(self, cohort_df, source=False):
        tempTableName = '##' + str(uuid.uuid4()).split('-')[0]
        cohort_df.to_sql(tempTableName, con=self.engine, index=False, if_exists='replace')
        if not source:
            sql = '''
                SELECT 
                    cohort.person_id, 
                    person.birth_datetime,
                    person.gender_concept_id,
                    c1.concept_name as gender_concept_name,
                    person.race_concept_id,
                    c2.concept_name as race_concept_name,
                    person.ethnicity_concept_id,
                    c3.concept_name as ethnicity_concept_name
                FROM {tempTableName} cohort
                LEFT JOIN person
                ON cohort.person_id = person.person_id
                LEFT JOIN concept c1
                ON c1.concept_id = person.gender_concept_id
                LEFT JOIN concept c2
                ON c2.concept_id = person.race_concept_id
                LEFT JOIN concept c3
                ON c3.concept_id = person.ethnicity_concept_id
            '''.format(tempTableName=tempTableName)
            df = pd.read_sql(sql,self.cnxn)
        else:
            sql = '''
                SELECT 
                    person.*
                FROM {tempTableName} cohort
                LEFT JOIN person
                ON cohort.person_id = person.person_id
            '''.format(tempTableName=tempTableName)
            df = pd.read_sql(sql,self.cnxn)

        sql = '''
                DROP TABLE {t}
                '''.format(t = tempTableName)
        self.cursor.execute(sql)
        self.cursor.commit()
        return df

    def get_visit(self, cohort_df, source=False, visit_concept_id=['9201','9202','9203']):
        tempTableName = '##' + str(uuid.uuid4()).split('-')[0]
        cohort_df.to_sql(tempTableName, con=self.engine, index=False, if_exists='replace')
        if visit_concept_id is None:
            # return all visits.
            where_clause = ''
        else:
            where_clause = '''
                WHERE visit_occurrence.visit_concept_id IN ({visit_concept_id_list})
            '''.format(visit_concept_id_list=','.join(visit_concept_id))
        if not source:
            sql = '''
                SELECT 
                    cohort.person_id, 
                    visit_occurrence.visit_concept_id,
                    c1.concept_name AS visit_concept_name,
                    visit_occurrence.visit_start_date,
                    visit_occurrence.visit_end_date,
                    visit_occurrence.visit_occurrence_id
                FROM {tempTableName} cohort
                LEFT JOIN visit_occurrence
                ON cohort.person_id = visit_occurrence.person_id
                LEFT JOIN concept c1
                ON c1.concept_id = visit_occurrence.visit_concept_id
                {where_clause}
            '''.format(tempTableName=tempTableName, where_clause=where_clause)
            df = pd.read_sql(sql,self.cnxn)
        else:
            sql = '''
                SELECT 
                    visit_occurrence.*
                FROM {tempTableName} cohort
                LEFT JOIN visit_occurrence
                ON cohort.person_id = visit_occurrence.person_id
                {where_clause}
            '''.format(tempTableName=tempTableName, where_clause=where_clause)
            df = pd.read_sql(sql,self.cnxn)
        sql = '''
                DROP TABLE {t}
                '''.format(t = tempTableName)
        self.cursor.execute(sql)
        self.cursor.commit()
        return df

    def get_condition(self, cohort_df, source=False, condition_concept_id=None):
        tempTableName = '##' + str(uuid.uuid4()).split('-')[0]
        cohort_df.to_sql(tempTableName, con=self.engine, index=False, if_exists='replace')
        if condition_concept_id is None:
            # return all visits.
            where_clause = ''
        else:
            condition_concept_id = '''
                WHERE condition_occurrence.condition_concept_id IN ({condition_concept_id_list})
            '''.format(condition_concept_id_list=','.join(condition_concept_id))
        if not source:
            sql = '''
                SELECT 
                    DISTINCT
                    cohort.person_id, 
                    condition_occurrence.visit_occurrence_id,
                    condition_occurrence.condition_start_date,
                    condition_occurrence.condition_end_date,
                    condition_occurrence.condition_concept_id,
                    c1.concept_name AS condition_concept_name
                FROM {tempTableName} cohort
                LEFT JOIN condition_occurrence
                ON cohort.person_id = condition_occurrence.person_id
                LEFT JOIN visit_occurrence
                ON condition_occurrence.visit_occurrence_id = visit_occurrence.visit_occurrence_id
                LEFT JOIN concept c1
                ON c1.concept_id = condition_occurrence.condition_concept_id
                {where_clause}
            '''.format(tempTableName=tempTableName, where_clause=where_clause)
            df = pd.read_sql(sql,self.cnxn)
        
        else:
            sql = '''
                SELECT 
                    condition_occurrence.*
                FROM {tempTableName} cohort
                LEFT JOIN condition_occurrence
                ON cohort.person_id = condition_occurrence.person_id
                {where_clause}
            '''.format(tempTableName=tempTableName, where_clause=where_clause)
            df = pd.read_sql(sql,self.cnxn)

        sql = '''
                DROP TABLE {t}
                '''.format(t = tempTableName)
        self.cursor.execute(sql)
        self.cursor.commit()
        return df

    def get_drug(self, cohort_df, source=False, drug_concept_id=None):
        tempTableName = '##' + str(uuid.uuid4()).split('-')[0]
        cohort_df.to_sql(tempTableName, con=self.engine, index=False, if_exists='replace')
        if drug_concept_id is None:
            # return all visits.
            where_clause = ''
        else:
            drug_concept_id = '''
                WHERE drug_exposure.drug_concept_id IN ({drug_concept_id_list})
            '''.format(drug_concept_id_list=','.join(drug_concept_id))
        if not source:
            sql = '''
                SELECT 
                    DISTINCT
                    cohort.person_id, 
                    drug_exposure.visit_occurrence_id,
                    drug_exposure.drug_exposure_start_date,
                    drug_exposure.drug_exposure_end_date,
                    drug_exposure.drug_concept_id,
                    c1.concept_name AS drug_concept_name
                FROM {tempTableName} cohort
                LEFT JOIN drug_exposure
                ON cohort.person_id = drug_exposure.person_id
                LEFT JOIN visit_occurrence
                ON drug_exposure.visit_occurrence_id = visit_occurrence.visit_occurrence_id
                LEFT JOIN concept c1
                ON c1.concept_id = drug_exposure.drug_concept_id
                {where_clause}
            '''.format(tempTableName=tempTableName, where_clause=where_clause)
            df = pd.read_sql(sql,self.cnxn)
        
        else:
            sql = '''
                SELECT 
                    drug_exposure.*
                FROM {tempTableName} cohort
                LEFT JOIN drug_exposure
                ON cohort.person_id = drug_exposure.person_id
                {where_clause}
            '''.format(tempTableName=tempTableName, where_clause=where_clause)
            df = pd.read_sql(sql,self.cnxn)

        sql = '''
                DROP TABLE {t}
                '''.format(t = tempTableName)
        self.cursor.execute(sql)
        self.cursor.commit()
        return df

    def get_procedure(self, cohort_df, source=False, procedure_concept_id=None):
        tempTableName = '##' + str(uuid.uuid4()).split('-')[0]
        cohort_df.to_sql(tempTableName, con=self.engine, index=False, if_exists='replace')
        if procedure_concept_id is None:
            # return all visits.
            where_clause = ''
        else:
            procedure_concept_id = '''
                WHERE procedure_occurrence.procedure_concept_id IN ({procedure_concept_id_list})
            '''.format(procedure_concept_id_list=','.join(procedure_concept_id))
        if not source:
            sql = '''
                SELECT 
                    DISTINCT
                    cohort.person_id, 
                    procedure_occurrence.visit_occurrence_id,
                    procedure_occurrence.procedure_date,
                    procedure_occurrence.procedure_concept_id,
                    c1.concept_name AS procedure_concept_name
                FROM {tempTableName} cohort
                LEFT JOIN procedure_occurrence
                ON cohort.person_id = procedure_occurrence.person_id
                LEFT JOIN visit_occurrence
                ON procedure_occurrence.visit_occurrence_id = visit_occurrence.visit_occurrence_id
                LEFT JOIN concept c1
                ON c1.concept_id = procedure_occurrence.procedure_concept_id
                {where_clause}
            '''.format(tempTableName=tempTableName, where_clause=where_clause)
            df = pd.read_sql(sql,self.cnxn)
        
        else:
            sql = '''
                SELECT 
                    procedure_occurrence.*
                FROM {tempTableName} cohort
                LEFT JOIN procedure_occurrence
                ON cohort.person_id = procedure_occurrence.person_id
                {where_clause}
            '''.format(tempTableName=tempTableName, where_clause=where_clause)
            df = pd.read_sql(sql,self.cnxn)

        sql = '''
                DROP TABLE {t}
                '''.format(t = tempTableName)
        self.cursor.execute(sql)
        self.cursor.commit()
        return df
    
    def get_observation(self, cohort_df, source=False, observation_concept_id=None):
        tempTableName = '##' + str(uuid.uuid4()).split('-')[0]
        cohort_df.to_sql(tempTableName, con=self.engine, index=False, if_exists='replace')
        if observation_concept_id is None:
            # return all visits.
            where_clause = ''
        else:
            observation_concept_id = '''
                WHERE observation.observation_concept_id IN ({observation_concept_id_list})
            '''.format(observation_concept_id_list=','.join(observation_concept_id))
        if not source:
            sql = '''
                SELECT 
                    DISTINCT
                    cohort.person_id, 
                    observation.visit_occurrence_id,
                    observation.observation_date,
                    observation.observation_concept_id,
                    c1.concept_name AS observation_concept_name,
                    observation.value_as_number,
                    observation.value_as_string,
                    observation.value_as_concept_id,
                    c2.concept_name AS value_concept_name,
                    observation.qualifier_concept_id,
                    c3.concept_name AS qualifier_concept_name,
                    observation.unit_concept_id,
                    c4.concept_name AS unit_concept_name
                FROM {tempTableName} cohort
                LEFT JOIN observation
                ON cohort.person_id = observation.person_id
                LEFT JOIN visit_occurrence
                ON observation.visit_occurrence_id = visit_occurrence.visit_occurrence_id
                LEFT JOIN concept c1
                ON c1.concept_id = observation.observation_concept_id
                LEFT JOIN concept c2
                ON c2.concept_id = observation.value_as_concept_id
                LEFT JOIN concept c3
                ON c3.concept_id = observation.qualifier_concept_id
                LEFT JOIN concept c4
                ON c4.concept_id = observation.unit_concept_id
                {where_clause}
            '''.format(tempTableName=tempTableName, where_clause=where_clause)
            df = pd.read_sql(sql,self.cnxn)
        
        else:
            sql = '''
                SELECT 
                    observation.*
                FROM {tempTableName} cohort
                LEFT JOIN observation
                ON cohort.person_id = observation.person_id
                {where_clause}
            '''.format(tempTableName=tempTableName, where_clause=where_clause)
            df = pd.read_sql(sql,self.cnxn)

        sql = '''
                DROP TABLE {t}
                '''.format(t = tempTableName)
        self.cursor.execute(sql)
        self.cursor.commit()
        return df
    
    def get_measurement(self, cohort_df, source=False, measurement_concept_id=None):
        tempTableName = '##' + str(uuid.uuid4()).split('-')[0]
        cohort_df.to_sql(tempTableName, con=self.engine, index=False, if_exists='replace')
        if measurement_concept_id is None:
            # return all visits.
            where_clause = ''
        else:
            measurement_concept_id = '''
                WHERE measurement.measurement_concept_id IN ({measurement_concept_id_list})
            '''.format(measurement_concept_id_list=','.join(measurement_concept_id))
        if not source:
            sql = '''
                SELECT 
                    DISTINCT
                    cohort.person_id, 
                    measurement.visit_occurrence_id,
                    measurement.measurement_date,
                    measurement.measurement_concept_id,
                    c1.concept_name AS measurement_concept_name,
                    measurement.value_as_number,
                    measurement.value_as_concept_id,
                    c2.concept_name AS value_concept_name,
                    measurement.unit_concept_id,
                    c3.concept_name AS unit_concept_name
                FROM {tempTableName} cohort
                LEFT JOIN measurement
                ON cohort.person_id = measurement.person_id
                LEFT JOIN visit_occurrence
                ON measurement.visit_occurrence_id = visit_occurrence.visit_occurrence_id
                LEFT JOIN concept c1
                ON c1.concept_id = measurement.measurement_concept_id
                LEFT JOIN concept c2
                ON c2.concept_id = measurement.value_as_concept_id
                LEFT JOIN concept c3
                ON c3.concept_id = measurement.unit_concept_id
                {where_clause}
            '''.format(tempTableName=tempTableName, where_clause=where_clause)
            df = pd.read_sql(sql,self.cnxn)
        
        else:
            sql = '''
                SELECT 
                    measurement.*
                FROM {tempTableName} cohort
                LEFT JOIN measurement
                ON cohort.person_id = measurement.person_id
                {where_clause}
            '''.format(tempTableName=tempTableName, where_clause=where_clause)
            df = pd.read_sql(sql,self.cnxn)

        sql = '''
                DROP TABLE {t}
                '''.format(t = tempTableName)
        self.cursor.execute(sql)
        self.cursor.commit()
        return df


    @staticmethod
    def get_master_df(cohort_df, demo_df, visit_df, condition_df, measurement_df, procedure_df):
        master_df = cohort_df.merge(demo_df, on='person_id', how='left')
        master_df = master_df.merge(visit_df, on='person_id', how='left')
        master_df = master_df.merge(condition_df, on=['person_id','visit_occurrence_id'], how='left')
        master_df = master_df.merge(measurement_df, on=['person_id','visit_occurrence_id'], how='left')
        master_df = master_df.merge(procedure_df, on=['person_id','visit_occurrence_id'], how='left')
        master_df['age_at_visit'] = (pd.to_datetime(master_df['visit_start_date']) - master_df['birth_datetime']).dt.days / 365

        def lable_group(row):
            if row['age_at_visit'] < 15:
                return '0'
            if row['age_at_visit'] < 18:
                return '1-ped'
            elif row['age_at_visit'] < 21:
                return '2-transit'
            elif row['age_at_visit'] < 24:
                return '3-adult'
            else:
                return '4'
        master_df['group'] = master_df.apply(lambda x: lable_group(x), axis=1)
        return master_df

    @staticmethod
    def get_agg_df(master_df, demo_df):
        event_count_df = master_df.groupby(['group','person_id'])[['condition_concept_id','measurement_concept_id','procedure_concept_id']].nunique().reset_index()
        event_count_df.columns = ['group','person_id','conditions','measurements','procedures']
        observation_period_days_df = master_df.groupby(['group','person_id'])[['visit_start_date']].agg(lambda x: (max(x) - min(x)).days + 1).reset_index()
        observation_period_days_df.columns = ['group','person_id','observation_period_days']
        visit_type_df = master_df.groupby(['group','person_id','visit_concept_id'])['visit_occurrence_id'].nunique().reset_index()
        visit_type_df = visit_type_df.pivot(index=['group','person_id'], columns='visit_concept_id', values=['visit_occurrence_id']).reset_index().fillna(0)
        visit_type_df.columns = ['group','person_id','inpatient_visits','outpatient_visits','ER_visits']
        agg_df = event_count_df.merge(observation_period_days_df, on=['group','person_id'], how='left').merge(visit_type_df, on=['group','person_id'], how='left').fillna(0)
        agg_df = agg_df.merge(demo_df, on='person_id', how='left')
        return agg_df

    @staticmethod
    def get_pivot_df(df, column, required_num_outpatient_visits=0, required_observation_period_days=0):
        pd.set_option('float_format', '{:.2f}'.format)
        # warnings.filterwarnings('ignore')
        df = df[df['outpatient_visits']>=required_num_outpatient_visits]
        df = df[df['observation_period_days']>=required_observation_period_days]
        subset_df = df[['person_id','group','birth_datetime','observation_period_days',column]]
        subset_df['current_age'] = 2022 - subset_df['birth_datetime'].dt.year
        def get_total_years(row):
            if row['group'] == '1-ped':
                return 3
            elif row['group'] == '2-transit':
                return 3
            elif row['group'] == '3-adult':
                return 3
            else: 
                return row['current_age'] - 21
        subset_df['total_years'] = subset_df.apply(lambda x: get_total_years(x), axis=1)
        subset_df['Total'] = subset_df[column]
        subset_df['Annual'] = subset_df['Total']/subset_df['total_years']
        subset_df['Annual (Norm)'] = subset_df['Total']/subset_df['observation_period_days'] * 365
        subset_df = subset_df[subset_df['group'].isin(['1-ped','2-transit','3-adult'])]
        pivot_df = subset_df.pivot_table(index='person_id', columns = 'group',values=['Annual','Annual (Norm)']).dropna().reset_index().drop([('person_id','')], axis=1)
        return pivot_df
        # selected_description = ['25%','50%','75%']
        # statistics = ['Total','Annual','Annual (Norm)']
        # selected_columns = [('group','')]
        # selected_columns.extend([(column,statistic)  for column in statistics for statistic in selected_description])
        # summary_df = subset_df.groupby(['group'])[statistics].describe().reset_index()[selected_columns]
        # return summary_df

    @staticmethod
    def get_summary(pivot_df):
        df = pivot_df.apply(lambda x: x.describe(), axis=0)
        df = df.iloc[4:7,:]       
        return df

    @staticmethod
    def plot_connected_figure(pivot_df):
        import matplotlib.pyplot as plt
        groups = pivot_df.columns.get_level_values(None).unique()  # Get unique groups

        for group in groups:
            fig, ax = plt.subplots()
            ax.set_title(f'{group}')
            ax.set_xlabel('Group')
            ax.set_ylabel('Value')

            # Extract data for the current group
            group_data = pivot_df[group]

            # Plot lines connecting the three points for each individual
            for index, row in group_data.iterrows():
                values = row.values[-1]
                ax.plot(range(1, 3), values[-1], marker='o')

            ax.set_xticks(range(1, 3))
            ax.set_xticklabels(['1-ped', '2-transit', '3-adult'])

        plt.show()


    @staticmethod
    def get_follow_up_pattern(df, required_num_outpatient_visits=0, required_observation_period_days=0):
        subset_df = df[['group','person_id','outpatient_visits','observation_period_days']]
        subset_df = subset_df[subset_df['outpatient_visits']>=required_num_outpatient_visits]
        subset_df = subset_df[subset_df['observation_period_days']>=required_observation_period_days]
        summary_df = subset_df.groupby('group')['person_id'].nunique().reset_index()
        return summary_df


class SolrManager():
    def __init__(self,configFile='/projects/phi/cl3720/db.conf'):
        self.config = configparser.ConfigParser()
        self.config.read('/projects/phi/cl3720/db.conf')
        # self.config.sections()
        self.solrhost = self.config['SOLR']['solrhost']
        self.username = self.config['SOLR']['username']
        self.password = self.config['SOLR']['password']
        qt = "select"
        self.solr = pysolr.Solr(self.solrhost, search_handler="/"+qt, always_commit=True, timeout=20, auth=(self.username,self.password))

    def getSolr(self):
        # cursor = cnxn.cursor()
        return self.solr
    
    def refreshSolrConnection(self,timeout=100):
        qt = "select"
        self.solr = pysolr.Solr(self.solrhost, search_handler="/"+qt, always_commit=True, timeout=100, auth=(self.username,self.password))

    def get_note(self, empi, source = False, meta_only=False, keywords=None,title=None, is_scanned_text=None, provider_name=None, start_date=None, end_date=None):

        q = f'''empi: ({empi})'''
        
        # e.g. title = ["sPEDS Genetics", "consult visit"]
        if title is not None and meta_only:
            q = q + ' AND ' + '(' + ' OR '.join(['(' + ' AND '.join(['(title: {to}~'.format(to=to) + ')' for to in ti.split(' ') if len(to)>2]) + ')' for ti in title]) + ')'
        
        if (is_scanned_text is not None and keywords is None) and meta_only:
            q = q + ' AND ' + '(' + 'is_scanned_text : {is_scanned_text}'.format(is_scanned_text=is_scanned_text) + ')'

        if provider_name is not None and meta_only:
            q = q + ' AND ' + '(' + ' OR '.join(['(' + ' AND '.join(['(provider_name: {to}~'.format(to=to) + ')' for to in ti.split(' ') if len(to)>2]) + ')' for ti in provider_name]) + ')'
        
        if ((start_date is not None) or (end_date is not None)) and meta_only:
            if start_date is None: 
                start_date = '*'
            if end_date is None:
                end_date = '*'
            q = q + ' AND ' + f'primary_time: [{start_date} TO {end_date}]'

        if keywords is not None and meta_only:
            if is_scanned_text is not None:
                raise "Error: is_scanned_text is not valid while keyword is provided."
            q = q + ' AND ' + '(' + 'is_scanned_text : {is_scanned_text}'.format(is_scanned_text=False) + ')' # non scanned doc only.

            q = q + ' AND ' + '(' + ' OR '.join([f'text: "{k}"~' for k in keywords]) + ')'


        fl = ['primary_time', 'empi', 'patient_name','organization', 'event_code', 'event_status', 'cwid', 'provider_name', 'update_time', 'title', 'text_length', 'is_scanned_text', 'id', 'text']

        if meta_only:
            fl.remove('text')

        

        print(q)

        results = self.solr.search(q, **{
                'fl' : fl,
                'rows': 1
            })
        maxRows = results.raw_response['response']['numFound']
        
        # maxRows = 1000
        start = 0
        docs = []
        while(start < maxRows):
        # return 10000 per batch.
            try_flag = 1
            while(try_flag):
                if try_flag > 3:
                    sys.exit("Tried more than three times. Fatal Error can not be recovered.")
                try:
                    results = self.solr.search(q, **{
                        'fl' : fl,
                        'start' : start,
                        'rows': 100000
                    })
                    docs = docs + results.docs
                    start += 100000
                    try_flag = 0
                except:
                    try_flag += 1
                    self.refreshSolrConnection(timeout=500)
                    time.sleep(15)
        df = pd.DataFrame(docs)

        if not source:
            if df.shape[0] > 0:
                df['start_date'] = pd.to_datetime(df['primary_time']).dt.date
                df['end_date'] = pd.to_datetime(df['update_time']).dt.date
                columns = [i for i in ['empi', 'start_date','end_date', 'provider_name', 'title', 'is_scanned_text', 'text', 'id'] if i in fl or i in ['start_date','end_date']]
                df = df[columns]
        return df


    def _generateQuery(self,term:str, includeNegation=False):
        if includeNegation:
            q = '''TEXT.string:("{0}"~{1})'''.format(term,3)
            return(q)
        else:
            negexTriggers = ['no','negative','without','deny','cannot','never','exclude','rule out','unlike']
            initQ = ['''("{0}"~{1})'''.format(term,3)]
            for nTerm in negexTriggers:
                nQ = '''("{0}"~{1})'''.format(nTerm + ' ' + term,4)
                initQ.append(nQ)
            q = ' NOT '.join(initQ)
            q = 'TEXT.string: (' + q + ')'
            return(q)

    def _parseRawSolrResponse(self,docs:dict):
        patientDf = pd.DataFrame.from_records(docs,columns={'PRIMARY_TIME.long','EMPI.string','TITLE.string'})
        # filter out None
        patientDf = patientDf.dropna()
        # filter screened files start with 's'
        patientDf['TITLE'] = patientDf['TITLE.string'].apply(lambda x: x[0])
        patientDf[~patientDf['TITLE'].str.startswith('s')]
        # convert time to timestamp
        patientDf['PRIMARY_TIME'] = patientDf['PRIMARY_TIME.long'].apply(lambda x: self._convertTimestamp(x[0]))
        # convert EMPI to float
        patientDf['EMPI'] = patientDf['EMPI.string'].apply(lambda x: self._convertFloat(x[0]))
        return(patientDf[['EMPI','PRIMARY_TIME','TITLE']])

    def _convertTimestamp(self,x):
        try:
            return datetime.datetime.fromtimestamp(x/1000).date()
        except:
            return None

    def _convertFloat(self,x):
        if math.isnan(x):
            return None
        else:
            return '{:.0f}'.format(x)

    def getAllMeta(self):
        q = '''*.*'''
        results = self.solr.search(q, **{
                'fl' : "EMPI.string",
                'rows': 1
            })
        maxRows = results.raw_response['response']['numFound']
        logging.error('max row:  {0}'.format(maxRows))
        # maxRows = 1000
        start = 0
        docs = []
        while(start < maxRows):
        # return 10000 per batch.
            try_flag = 1
            while(try_flag):
                if try_flag > 3:
                    sys.exit("Tried more than three times. Fatal Error can not be recovered.")
                try:
                    logging.error('return {0} - {1}'.format(start,start+100000))
                    results = self.solr.search(q, **{
                        'fl' : ["EMPI.string","PRIMARY_TIME.long","TITLE.string","PROVIDER_NAME.string","TEXT.string","id"],
                        'start' : start,
                        'rows': 100000
                    })
                    docs = docs + results.docs
                    start += 100000
                    try_flag = 0
                except:
                    try_flag += 1
                    logging.error('solr connection error!')
                    self.refreshSolrConnection(timeout=500)
                    time.sleep(15)


        patientDf = pd.DataFrame.from_records(docs,columns={"EMPI.string","PRIMARY_TIME.long","TITLE.string","PROVIDER_NAME.string","TEXT.string","id"})
        # filter out None
        patientDf = patientDf.dropna()
        # filter screened files start with 's'
        patientDf['TITLE'] = patientDf['TITLE.string'].apply(lambda x: x[0])
        # patientDf[~patientDf['TITLE'].str.startswith('s')]
        # convert time to timestamp
        patientDf['PRIMARY_TIME'] = patientDf['PRIMARY_TIME.long'].apply(lambda x: self._convertTimestamp(x[0]))
        # convert EMPI to float
        patientDf['EMPI'] = patientDf['EMPI.string'].apply(lambda x: self._convertFloat(x[0]))
        patientDf['PROVIDER'] = patientDf['PROVIDER_NAME.string'].apply(lambda x: x[0])
        patientDf['TEXT_LENGTH'] = patientDf['TEXT.string'].apply(lambda x: len(x[0]))
        return patientDf[['id','TITLE','PRIMARY_TIME','EMPI','PROVIDER','TEXT_LENGTH']]
    
    def querySolr(self,term:str, maxRowOnly=True):
        q = self._generateQuery(term) 
        results = self.solr.search(q, **{
            'fl' : "EMPI.string",
            'rows': 1
        })
        maxRows = results.raw_response['response']['numFound']
        logging.warning('The total record returned for term [{0}]: [{1}]'.format(term,maxRows))
        if maxRowOnly:
            return pd.DataFrame()
        else:
            if maxRows > int(0.01 * 10000000): # assume a useless feature due to occured too often.
                logging.warning('Too many records returned for term [{0}] - Skip this term'.format(term))
                return pd.DataFrame()
            else:
                start = 0
                docs = []
                while(start < maxRows):
                    # return 10000 per batch.
                    results = self.solr.search(q, **{
                        'fl' : ["EMPI.string","PRIMARY_TIME.long","TITLE.string"],
                        'start' : start,
                        'rows': 100000
                    })
                    start += 100000
                    docs = docs + results.docs
                patientDf = self._parseRawSolrResponse(docs)    
                return patientDf

    def getPatientByTermList(self,termList:list,idManager):
        
        patientSolrDf = pd.DataFrame() 
        for term in termList:
            termDf = self.querySolr(term,maxRowOnly=False)
            termDf['term'] = term
            patientSolrDf = patientSolrDf.append(termDf)
        myidManager = IdManager(type='epic')
        myidManager.addIdList(patientSolrDf['EMPI'].to_list())
        myidManager.getAllIds()
        patientSolrDf = patientSolrDf.merge(myidManager.IdMappingDf[['EMPI','person_id']].drop_duplicates(),how='left')
        # standadize to patient view
        patientSolrDf['event'] = patientSolrDf['term']
        patientSolrDf['event_group'] = patientSolrDf['term']
        patientSolrDf['domain'] = 'solr_' + patientSolrDf['TITLE']
        patientSolrDf['year_of_event'] = patientSolrDf['PRIMARY_TIME'].astype(str).str.split('-').str[0]
        patientSolrDf = patientSolrDf[['person_id','event','event_group','year_of_event','domain']].drop_duplicates()
        logging.warning('[{y}] patients returned by querying solr'.format(y = str(len(set(patientSolrDf['person_id'].tolist())))))
        return patientSolrDf

class IdManager():
    def __init__(self, type,configFile='/projects/phi/cl3720/db.conf',database='ohdsi_cumc_2021q3r1'):
        self.sqlconnector = SqlConnector(configFile,database)
        self.engine = self.sqlconnector.getEngine()
        self.cnxn, self.cursor = self.sqlconnector.getCnxn()
        if type not in ['epic','mrn','nyp','crown','person_id']:
            raise ValueError('''
            IdManager only supports the following types:
                1. epic: Epic ID or EMPI;
                2. mrn: this is a vagor term. Both crown and nyp Ids will be searched and returned;
                3. nyp: NYP Id;
                4. crown: Outpaitient Crown Id;
                5. person_id: OHDSI person_id;
            ''')
        self.type = type
        self.inputIdList = []
        self.IdMappingDf = None
    
    def addIdList(self,idList):
        self.inputIdList = list(idList)
    
    def getAllIds(self):
        if self.type.lower() == 'epic':
            sql = '''
                SELECT DISTINCT M.person_id, M.EMPI, M.LOCAL_PT_ID, M.FACILITY_CODE
                FROM [mappings].[patient_mappings] M 
                where M.LOCAL_PT_ID IN ({l}) AND M.FACILITY_CODE = 'UI'
            '''.format(l = ','.join([ "'" + str(p) + "'" for p in self.inputIdList]))
            self.IdMappingDf = pd.read_sql(sql,self.cnxn)
            return 1
        
        if self.type.lower() == 'mrn':
            sql = '''
                SELECT DISTINCT M.person_id, M.EMPI, M.LOCAL_PT_ID, M.FACILITY_CODE
                FROM [mappings].[patient_mappings] M 
                where M.LOCAL_PT_ID IN ({l}) 
            '''.format(l = ','.join([ "'" + str(p) + "'" for p in self.inputIdList]))
            self.IdMappingDf = pd.read_sql(sql,self.cnxn)
            return 1

        if self.type.lower() == 'nyp':
            sql = '''
                SELECT DISTINCT M.person_id, M.EMPI, M.LOCAL_PT_ID, M.FACILITY_CODE
                FROM [mappings].[patient_mappings] M 
                where M.LOCAL_PT_ID IN ({l}) AND M.FACILITY_CODE = 'P'
            '''.format(l = ','.join([ "'" + str(p) + "'" for p in self.inputIdList]))
            self.IdMappingDf = pd.read_sql(sql,self.cnxn)
            return 1
        
        if self.type.lower() == 'crown':
            sql = '''
                SELECT DISTINCT M.person_id, M.EMPI, M.LOCAL_PT_ID, M.FACILITY_CODE
                FROM [mappings].[patient_mappings] M 
                where M.LOCAL_PT_ID IN ({l}) AND M.FACILITY_CODE = 'A'
            '''.format(l = ','.join([ "'" + str(p) + "'" for p in self.inputIdList]))
            self.IdMappingDf = pd.read_sql(sql,self.cnxn)
            return 1
        
        if self.type.lower() == 'person_id':
            sql = '''
                SELECT DISTINCT M.person_id, M.EMPI, M.LOCAL_PT_ID, M.FACILITY_CODE
                FROM [mappings].[patient_mappings] M 
                where M.person_id IN ({l})
            '''.format(l = ','.join([ "'" + str(p) + "'" for p in self.inputIdList]))
            self.IdMappingDf = pd.read_sql(sql,self.cnxn)
            return 1
