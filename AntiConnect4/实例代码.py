def alpha_beta(self, depth, alpha, beta):  # alpha-beta剪枝，alpha是大可能下界，beta是最小可能上界
    who = (self.max_depth - depth) % 2  # 那个玩家
    if self.is_game_over(who):  # 判断是否游戏结束，如果结束了就不用搜了
        return cc.min_val
    if depth == 1:  # 搜到指定深度了，也不用搜了
        # print(self.evaluate(who))
        return self.evaluate(who)
    move_list = self.board.generate_move(who)  # 返回所有能走的方法
    # 利用历史表0
    for i in range(len(move_list)):
        move_list[i].score = self.history_table.get_history_score(who, move_list[i])
    move_list.sort()  # 为了让更容易剪枝利用历史表得分进行排序
    best_step = move_list[0]
    score_list = []
    for step in move_list:
        temp = self.move_to(step)
        score = -self.alpha_beta(depth - 1, -beta, -alpha)  # 因为是一层选最大一层选最小，所以利用取负号来实现
        score_list.append(score)
        self.undo_move(step, temp)
        if score > alpha:
            alpha = score
            if depth == self.max_depth:
                self.best_move = step
            best_step = step
        if alpha >= beta:
            best_step = step
            break
    # print(score_list)
    # 更新历史表
    if best_step.from_x != -1:
        self.history_table.add_history_score(who, best_step, depth)
    return alpha
