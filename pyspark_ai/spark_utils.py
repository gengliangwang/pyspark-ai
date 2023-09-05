from pyspark import Row
from pyspark.errors import PySparkException
from pyspark.sql import SparkSession, DataFrame


class SparkUtils:
    def __init__(self, spark: SparkSession):
        self.spark = spark

    @staticmethod
    def _convert_row_as_tuple(row: Row) -> tuple:
        return tuple(map(str, row.asDict().values()))

    def _get_dataframe_results(self, df: DataFrame) -> list:
        return list(map(self._convert_row_as_tuple, df.collect()))

    def _run_command(self, command: str) -> str:
        df = self.spark.sql(command)
        return str(self._get_dataframe_results(df))

    def ask_spark(self, query: str):
        try:
            return self._run_command(query)
        except PySparkException as e:
            """Format the error message"""
            return f"Error: {e}"

    # Return functions for OpenAI Chat completion
    @staticmethod
    def get_functions(view_name, schema):
        return [
            {
                "name": "get_intermediate_results",
                "description": "If the task can't be finished with one sql query, e.g. pivot with dynamic value, "
                               "use this function to get intermediate results from Spark."
                               "Otherwise, if the task can be finished with one sql query, don't use this function."
                               "Input should be a fully formed SQL query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"""
                                    SQL query extracting info to answer the user's question.
                                    SQL should be written with select from {view_name} which contains the following schema:
                                    {schema}
                                    The query should be returned in plain text, not in JSON.
                                    """,
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "validate_query",
                "description": "Validate the query before returning it as an answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"""
                                    SQL query extracting info to answer the user's question.
                                    SQL should be written with select from {view_name} which contains the following schema:
                                    {schema}
                                    The query should be returned in plain text, not in JSON.
                                    """,
                        }
                    },
                    "required": ["query"],
                }
            }
        ]
