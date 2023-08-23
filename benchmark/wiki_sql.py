import json
import re
from argparse import ArgumentParser

from pyspark.sql import SparkSession

from pyspark_ai import SparkAI


def generate_sql_statements(table_file):
    sql_statements = []

    with open(table_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())

            table_name = item['id']
            # quote the headers with backticks
            headers = ["`{}`".format(h) for h in item['header']]
            header_str = "(" + ",".join(headers) + ")"
            rows = item['rows']

            values_str_list = []

            for row in rows:
                # Convert each value in the row to a string and escape single quotes
                values_str_list.append("(" + ",".join(["'{}'".format(str(val).replace("'", "''")) for val in row]) + ")")

            values_str = ",".join(values_str_list)
            create_statement = f"CREATE TEMP VIEW `{table_name}` AS SELECT * FROM VALUES {values_str} as {header_str};"
            sql_statements.append(create_statement)

    return sql_statements


# Parse questions and tables from the source file
def get_tables_and_questions(source_file):
    tables = []
    questions = []
    with open(source_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            tables.append(item['table_id'])
            questions.append(item['question'])
    return tables, questions


def convert_like_to_equal(sql_clause):
    # Remove surrounding whitespaces
    sql_clause = sql_clause.strip()

    # Check if the clause starts with WHERE and contains LIKE
    if ' LIKE ' in sql_clause:
        # Replace LIKE with =
        sql_clause = sql_clause.replace(' LIKE ', ' = ')

        # Replace % at the start and end of the value
        sql_clause = sql_clause.replace("'%", "'").replace("%'", "'")

        # Handle cases where % might be in the middle of the string, in which case it should be left as is
        sql_clause = sql_clause.replace("%%", "%")

    return sql_clause

def convert_to_wikisql_format(sql_query, table_schema):
    # Define aggregation and condition operations
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']

    # Initialize the WikiSQL format dictionary
    wikisql_format = {
        "query": {
            "sel": -1,
            "conds": [],
            "agg": 0
        },
        "error": ""
    }

    # Determine the selected column and aggregation
    agg_found = False
    for agg_op in agg_ops[1:]:
        # Two patterns: one expecting backticks, the other not
        pattern_with_backtick = f"{agg_op}\((DISTINCT\s*)?`([^`]+)`\)"
        pattern_without_backtick = f"{agg_op}\((DISTINCT\s*)?([^)]+)\)"

        match_with_backtick = re.search(pattern_with_backtick, sql_query)
        match_without_backtick = re.search(pattern_without_backtick, sql_query)

        if match_with_backtick:
            col_name = match_with_backtick.group(2).strip()
            wikisql_format["query"]["agg"] = agg_ops.index(agg_op)
            agg_found = True
            break
        elif match_without_backtick:
            col_name = match_without_backtick.group(2).strip()
            wikisql_format["query"]["agg"] = agg_ops.index(agg_op)
            agg_found = True
            break

    if not agg_found:
        # Remove DISTINCT if it appears right after SELECT
        refined_query = re.sub(r"SELECT\s+DISTINCT\s+", "SELECT ", sql_query)
        select_part = re.search("SELECT\s+(.+?)\s+FROM", refined_query).group(1)

        # if there is backtick in the first select part, get the column name from it
        if select_part.startswith("`"):
            col_name = select_part.split("`")[1]
        else:
            # if there is no backtick in the first select part,
            # get the column name from the first column
            col_name = select_part.split(",")[0].strip()

        wikisql_format["query"]["sel"] = table_schema.index(col_name)

    # Set the selected column index based on table schema
    wikisql_format["query"]["sel"] = table_schema.index(col_name)

    # Parse the WHERE clause for conditions
    where_match = re.search(r"WHERE\s+(.+)$", sql_query)
    if where_match:
        conditions = where_match.group(1).split("AND")
        for cond in conditions:
            cond = convert_like_to_equal(cond)
            col_name, op, value = re.search(r"(.+?)\s*([=><])\s*(.+)$", cond.strip()).groups()
            col_name = col_name.replace("`", "").strip()
            if value.isdigit():
                value = int(value)
            elif value.replace(".", "", 1).isdigit():
                value = float(value)
            else:
                value = value.strip("'")
            wikisql_format["query"]["conds"].append([table_schema.index(col_name), cond_ops.index(op), value])

    return wikisql_format


def split_columns_outside_backticks(s):
    # Split by commas but not those enclosed within backticks
    return re.split(r',(?=[^`]*((`[^`]*`)[^`]*`)*$)', s)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('table_file', help='table definition file')
    parser.add_argument('source_file', help='source file for the prediction')
    args = parser.parse_args()

    # Example usage:
    table_file = args.table_file
    statements = generate_sql_statements(table_file)
    spark = SparkSession.builder.getOrCreate()
    for stmt in statements:
        spark.sql(stmt)

    source_file = args.source_file
    tables, questions = get_tables_and_questions(source_file)
    spark_ai = SparkAI(spark_session=spark)
    # Create sql query for each question and table
    with open("pyspark_ai.jsonl", "w") as file:
        for table, question in zip(tables, questions):
            print(question)
            df = spark.table(f"`{table}`")
            sql_query = spark_ai._get_transform_sql_query(df, question, cache=True)
            spark_ai.commit()
            print(sql_query)
            wiki_sql_output = convert_to_wikisql_format(sql_query, df.columns)
            print(wiki_sql_output)
            file.write(json.dumps(wiki_sql_output) + "\n")

