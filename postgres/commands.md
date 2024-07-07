1. Copy data_schemas.sql to the docker container

```bash
docker cp postgres/data_schemas.sql insurance-fraud-modelling-db-1:/create_table.sql
```

2. Execute the create_table.sql

```bash
docker exec -it insurance-fraud-modelling-db-1 bash -c "psql -U username -d test -f /create_table.sql"
```

3. Copy datasets to docker

```bash
docker cp meaningful_df.csv insurance-fraud-modelling-db-1:/meaningful_df.csv         

docker cp ready_df.csv insurance-fraud-modelling-db-1:/ready_df.csv
```

4. Insert data into postgres

```bash
docker exec -it insurance-fraud-modelling-db-1 bash -c "psql -U username -d test -c \"\\copy meaningful_features FROM '/meaningful_df.csv' DELIMITER ',' CSV HEADER;\""

docker exec -it insurance-fraud-modelling-db-1 bash -c "psql -U username -d test -c \"\\copy model_data_w_dummy FROM '/ready_df.csv' DELIMITER ',' CSV HEADER;\""  
```