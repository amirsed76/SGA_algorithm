import sqlite3


class SqlManager:
    def __init__(self, file):
        self.conn = sqlite3.connect(file)
        self.crs = self.conn.cursor()
        self.create_database()

    def create_database(self):
        self.crs.execute("CREATE TABLE IF NOT EXISTS information ("
                         "fitness VARCHAR(100) NOT NULL  ,"
                         "max_gen INT NOT NULL ,"
                         "problem_size INT NOT NULL , "
                         "pop_size INT NOT NULL , "
                         "result VARCHAR(120) NOT NULL , "
                         "generation INT NOT NULL"
                         ")")

        self.crs.execute("DELETE FROM information")
        self.conn.commit()

    def add_row(self, fitness, max_gen, problem_size, pop_size, result, generation):
        sql = f"INSERT INTO information values ('{fitness}' , {max_gen} , {problem_size} , {pop_size} , '{result}' ,{generation} )"
        self.crs.execute(sql)
        self.conn.commit()
