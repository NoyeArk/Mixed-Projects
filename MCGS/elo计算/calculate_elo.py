import subprocess
import matplotlib.pyplot as plt


def gengrate_pgn(p1_name, p2_name, num_p1_win, num_p2_win):
    str_p1_win_p1_first = (
        f'[White "{p1_name}"]\n[Black "{p2_name}"]\n[Result "1-0"]\n1-0\n\n'
    )
    str_p1_win_p2_first = (
        f'[White "{p2_name}"]\n[Black "{p1_name}"]\n[Result "0-1"]\n0-1\n\n'
    )
    str_p2_win_p1_first = (
        f'[White "{p1_name}"]\n[Black "{p2_name}"]\n[Result "0-1"]\n0-1\n\n'
    )
    str_p2_win_p2_first = (
        f'[White "{p2_name}"]\n[Black "{p1_name}"]\n[Result "1-0"]\n1-0\n\n'
    )
    print(str_p2_win_p2_first)

    finally_str = ""
    for i in range(num_p1_win):
        if i % 2 == 0:
            finally_str += str_p1_win_p1_first
        else:
            finally_str += str_p1_win_p2_first
    for i in range(num_p2_win):
        if i % 2 == 0:
            finally_str += str_p2_win_p1_first
        else:
            finally_str += str_p2_win_p2_first
    print(finally_str)
    # 打开文件并写入字符串
    file_path = "1.pgn"  # 文件路径

    with open(file_path, "a") as file:
        file.write(finally_str)
    print("File created and content written successfully.")


def generate_all_pgn():
    # 交叉对战
    gengrate_pgn("minimax_500", "minimax_1000", 0, 120)
    gengrate_pgn("minimax_1000", "minimax_2000", 0, 120)
    gengrate_pgn("minimax_2000", "minimax_5000", 0, 120)
    gengrate_pgn("ab_500", "ab_1000", 0, 120)
    gengrate_pgn("ab_1000", "ab_2000", 0, 120)
    gengrate_pgn("ab_2000", "ab_5000", 0, 120)
    gengrate_pgn("mcts_500", "mcts_1000", 80, 100)
    gengrate_pgn("mcts_1000", "mcts_2000", 0, 120)
    gengrate_pgn("mcts_2000", "mcts_5000", 0, 120)
    gengrate_pgn("mcgs_500", "mcgs_1000", 90, 90)
    gengrate_pgn("mcgs_1000", "mcgs_2000", 59, 61)
    gengrate_pgn("mcgs_2000", "mcgs_5000", 90, 90)
    gengrate_pgn("gmcgs_500", "gmcgs_1000", 88, 92)
    gengrate_pgn("gmcgs_1000", "gmcgs_2000", 88, 92)
    gengrate_pgn("gmcgs_2000", "gmcgs_5000", 89, 91)

    # minimax vs ab
    gengrate_pgn("minimax_500", "ab_500", 0, 120)
    gengrate_pgn("minimax_1000", "ab_1000", 0, 120)
    gengrate_pgn("minimax_2000", "ab_2000", 0, 120)
    gengrate_pgn("minimax_5000", "ab_5000", 0, 120)
    # minimax vs mcts
    gengrate_pgn("minimax_500", "mcts_500", 0, 120)
    gengrate_pgn("minimax_1000", "mcts_1000", 0, 120)
    gengrate_pgn("minimax_2000", "mcts_2000", 0, 120)
    gengrate_pgn("minimax_5000", "mcts_5000", 0, 120)
    # minimax vs mcgs
    gengrate_pgn("minimax_500", "mcgs_500", 0, 120)
    gengrate_pgn("minimax_1000", "mcgs_1000", 0, 120)
    gengrate_pgn("minimax_2000", "mcgs_2000", 0, 120)
    gengrate_pgn("minimax_5000", "mcgs_5000", 0, 120)
    # minimax vs gmcgs
    gengrate_pgn("minimax_500", "gmcgs_500", 0, 120)
    gengrate_pgn("minimax_1000", "gmcgs_1000", 0, 120)
    gengrate_pgn("minimax_2000", "gmcgs_2000", 0, 120)
    gengrate_pgn("minimax_5000", "gmcgs_5000", 0, 120)

    # ab vs mcts
    gengrate_pgn("ab_500", "mcts_500", 0, 120)
    gengrate_pgn("ab_1000", "mcts_1000", 0, 120)
    gengrate_pgn("ab_2000", "mcts_2000", 0, 120)
    gengrate_pgn("ab_5000", "mcts_5000", 0, 120)
    # ab vs mcgs
    gengrate_pgn("ab_500", "mcgs_500", 0, 120)
    gengrate_pgn("ab_1000", "mcgs_1000", 0, 120)
    gengrate_pgn("ab_2000", "mcgs_2000", 0, 120)
    gengrate_pgn("ab_5000", "mcgs_5000", 0, 120)
    # ab vs gmcgs
    gengrate_pgn("ab_500", "gmcgs_500", 0, 120)
    gengrate_pgn("ab_1000", "gmcgs_1000", 0, 120)
    gengrate_pgn("ab_2000", "gmcgs_2000", 0, 120)
    gengrate_pgn("ab_5000", "gmcgs_5000", 0, 120)

    # mcts vs mcgs
    gengrate_pgn("mcts_500", "mcgs_500", 65, 55)
    gengrate_pgn("mcts_1000", "mcgs_1000", 63, 57)
    gengrate_pgn("mcts_2000", "mcgs_2000", 60, 60)
    gengrate_pgn("mcts_5000", "mcgs_5000", 57, 63)
    # mcts vs gmcgs
    gengrate_pgn("mcts_500", "gmcgs_500", 59, 61)
    gengrate_pgn("mcts_1000", "gmcgs_1000", 58, 62)
    gengrate_pgn("mcts_2000", "gmcgs_2000", 57, 63)
    gengrate_pgn("mcts_5000", "gmcgs_5000", 558, 642)

    # mcgs vs gmcgs
    gengrate_pgn("mcgs_500", "gmcgs_500", 60, 60)
    gengrate_pgn("mcgs_1000", "gmcgs_1000", 58, 62)
    gengrate_pgn("mcgs_2000", "gmcgs_2000", 85, 95)
    gengrate_pgn("mcgs_5000", "gmcgs_5000", 81, 99)

def generate_player_list(method_name, num_block, num_channel, total_generation):
    player_list = []
    for i in range(total_generation):
        player_list.append(f"{method_name}_{num_block}b{num_channel}f_V{i}")
    return player_list


def generate_all_player_list():
    # 6b128f, 10b128f, 20b128f, 10b256f, 20b256f
    all_players = ['minimax', 'ab', 'mcts', 'mcgs', 'gmcgs']
    # all_players.append(generate_player_list("dmc", 6, 128, 16))
    # all_players.append(generate_player_list("dmc", 10, 128, 16))
    # all_players.append(generate_player_list("dmc", 20, 128, 16))
    # all_players.append(generate_player_list("dmc", 10, 256, 16))
    # all_players.append(generate_player_list("dmc", 20, 256, 16))
    return all_players


def call_bayeselo():
    # 定义要调用的可执行文件路径
    exe_path = "bayeselo.exe"

    # 启动子进程并与其进行交互
    process = subprocess.Popen(
        exe_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    players_list = generate_all_player_list()
    # 向子进程发送输入
    input_data = "readpgn 1.pgn\nelo\n mm\n prediction\n  results\n"
    for i in range(len(players_list)):
        input_data += f"   addplayer {players_list[i]}\n"
    input_data += "  x\n  players;\n  x\nratings>elo.txt\n x\nx\n"

    print(input_data)
    process.stdin.write(input_data)
    process.stdin.flush()  # 确保输入被发送到子进程

    # 从子进程获取输出
    output, error = process.communicate()

    # 打印输出和错误信息
    print("Output:")
    print(output)


def read_elo():
    elo_list = [[], [], [], [], []]
    with open("elo.txt", "r") as f:
        line = f.readline()
        while line:
            tmp_line = line.split(" ")
            for it in tmp_line:
                if it == "":
                    tmp_line.remove(it)
            if tmp_line[1][0] != "N":
                class_name = tmp_line[1].split("_")[0]
                if class_name == "minimax":
                    elo_list[0].append([tmp_line[1], tmp_line[2]])
                elif class_name == "ab":
                    elo_list[1].append([tmp_line[1], tmp_line[2]])
                elif class_name == "mcts":
                    elo_list[2].append([tmp_line[1], tmp_line[2]])
                elif class_name == "mcgs":
                    elo_list[3].append([tmp_line[1], tmp_line[2]])
                elif class_name == "gmcgs":
                    elo_list[4].append([tmp_line[1], tmp_line[2]])
                else:
                    assert "overflow!"
            # print(tmp_line)
            line = f.readline()
    final_elo_list = get_sq_elo(elo_list)
    return final_elo_list


def get_sq_elo(elo_list):
    final_elo_list = [[], [], [], [], []]
    for k in range(len(elo_list)):
        index = []
        for i in range(len(elo_list[k])):
            index.append(i)
        generation = []
        for i in range(len(elo_list[k])):
            # generation.append(int(elo_list[k][i][0].split("V")[1]))
            generation.append(int(elo_list[k][i][1]))
        for i in range(len(generation)):
            min_num = generation[i]
            for j in range(i, len(generation)):
                if generation[j] < min_num:
                    min_num = generation[j]
                    tmp = generation[i]
                    generation[i] = generation[j]
                    generation[j] = tmp
                    tmp = index[i]
                    index[i] = index[j]
                    index[j] = tmp
        for i in range(len(index)):
            final_elo_list[k].append(elo_list[k][index[i]])
    return final_elo_list


def draw_elo_crave(final_elo_list):
    x = ['500', '1000', '2000', '5000']
    minimax = []
    ab = []
    mcts = []
    mcgs = []
    gmcgs = []
    for i in range(4):
        minimax.append(float(final_elo_list[0][i][1]) - float(final_elo_list[3][0][1]))
        ab.append(float(final_elo_list[1][i][1]) - float(final_elo_list[3][0][1]))
        mcts.append(float(final_elo_list[2][i][1]) - float(final_elo_list[3][0][1]))
        mcgs.append(float(final_elo_list[3][i][1]) - float(final_elo_list[3][0][1]))
        gmcgs.append(float(final_elo_list[4][i][1]) - float(final_elo_list[3][0][1]))
    # 创建图形对象和子图
    fig, ax = plt.subplots()

    # 绘制曲线
    ax.plot(x, minimax, label="minimax")
    ax.plot(x, ab, label="ab")
    ax.plot(x, mcts, label="mcts")
    ax.plot(x, mcgs, label="mcgs")
    ax.plot(x, gmcgs, label="gmcgs")

    ax.legend()
    # 设置图形标题和坐标轴标签
    ax.set_title("elo")
    ax.set_xlabel("state number")
    ax.set_ylabel("elo")

    # 显示图形
    plt.show()


if __name__ == "__main__":
    # generate_all_pgn()
    # call_bayeselo()
    final_elo_list = read_elo()
    draw_elo_crave(final_elo_list)
