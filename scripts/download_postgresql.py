import psycopg2
import pandas as pd
from configparser import ConfigParser

# Read configuration file (see https://neon.tech/postgresql/postgresql-python/connect)
filename, section = 'enrichedeurocrops.ini', 'postgresql'
parser = ConfigParser()
parser.read(filename)

config = {}
if parser.has_section(section):
    params = parser.items(section)
    for param in params:
        config[param[0]] = param[1]
    print(config)
else:
    raise Exception('Section {0} not found in the {1} file'.format(section, filename))

# Connect to AWS database
try:

    with psycopg2.connect(**config) as conn: # unpack dict
        print('Connected to the PostgreSQL server.')

        # Get table names
        query = "SELECT table_schema, table_name FROM information_schema.tables ORDER BY table_schema, table_name;"
        table_names = pd.read_sql_query(query, conn)
        print(table_names.table_name[ table_names.table_schema == 'public' ])
        # Relevant tables: TEST_parcels, geography_columns, geometry_columns, raster_columns, raster_overviews, spatial_ref_sys, v0p1.parcels

        # Query data
        query = "SELECT * FROM v0p1.parcels LIMIT 10;"
        data = pd.read_sql_query(query, conn)

        print(data)
except (psycopg2.DatabaseError, Exception) as error:
    print(error)

# # Export to CSV
# data.to_csv("dataset.csv", index=False)
# print("Dataset saved to dataset.csv")

conn.close()
