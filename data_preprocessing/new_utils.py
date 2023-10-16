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
    def __init__(self,configFile='/home/fc2718/rare_disease/db.conf',database='ohdsi_cumc_2022q4r1'):
        self.config = configparser.ConfigParser()
        self.config.read('/home/fc2718/rare_disease/db.conf')
        #print(self.config.sections())
        self.server = self.config['ELILEX']['server']
        self.database = database
        self.username = self.config['ELILEX']['username']
        self.password = self.config['ELILEX']['password']
        self.cnxn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};SERVER='+self.server+';DATABASE='+self.database+';UID='+self.username+';PWD='+ self.password)
        params = 'Driver={ODBC Driver 17 for SQL Server};SERVER='+self.server+';DATABASE='+self.database+';UID='+self.username+';PWD='+ self.password
        db_params = urllib.parse.quote_plus(params)
        url = "mssql+pyodbc:///?odbc_connect={}".format(db_params)
        print(url)
        self.engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect={}".format(db_params))
    def getCnxn(self):
        cursor = self.cnxn.cursor()
        return self.cnxn, cursor
    
    def getEngine(self):
        # cursor = cnxn.cursor()
        return self.engine
    


class OhdsiManager():
    def __init__(self,configFile='/home/fc2718/rare_disease/db.conf',database='ohdsi_cumc_2022q4r1'):
        self.sqlconnector = SqlConnector(configFile,database)
        self.engine = self.sqlconnector.getEngine()
        self.cnxn, self.cursor = self.sqlconnector.getCnxn()

    def get_cohort(self, top_x = 100, ancestor_concept_id = "138875", not_in = False, year_of_birth_start = 1980, year_of_birth_end = 2000, num_visits = 1, observation_period_days = 12):
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
    

    def get_dataFromQuery(self,sql_query):
        df = pd.read_sql(sql_query,self.cnxn)
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
            print("Acquire concept codes")
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



## Solr notes
class SolrManager():
    def __init__(self,configFile='/home/fc2718/rare_disease/db.conf'):
        self.config = configparser.ConfigParser()
        self.config.read('/home/fc2718/rare_disease/db.conf')
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

        
      
        fl = ['primary_time', 'empi', 'patient_name',  'organization', 'event_code', 'event_status', 'cwid', 'update_time', 'title', 'text_length', 'is_scanned_text', 'id', 'text']

        if meta_only:
            fl.remove('text')


        #print(q)

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
                columns = [i for i in ['empi', 'start_date','end_date' ,'title', 'is_scanned_text', 'text', 'id'] if i in fl or i in ['start_date','end_date']]
                df = df[columns]
            else: 
                return None

        return df
    
    def get_note_withProviders(self, empi, source = False, meta_only=False, keywords=None,title=None, is_scanned_text=None, provider_name=None, start_date=None, end_date=None):

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
            else:
                return None
            
        return df


class IdManager():
    def __init__(self, type,configFile='/home/fc2718/rare_disease/db.conf',database='ohdsi_cumc_2022q4r1'):
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




