// SPDX-FileCopyrightText: 2024 Henrik Jakob jakob@ibb.uni-stuttgart.de
// SPDX-License-Identifier: MIT

#pragma once

#include <spdlog/fmt/chrono.h>
#include <spdlog/fmt/fmt.h>
#include <unordered_map>

template <typename P = std::chrono::seconds>
class Timer
{
 public:
  using Clock = std::chrono::high_resolution_clock;
  using Time  = std::chrono::time_point<Clock>;
  using Period = P;


  void startTimer(std::string name) { startTimes.emplace(name, Clock::now()); }

  auto stopTimer(std::string name) {
    auto stopTime  = Clock::now();
    auto startTime = startTimes.at(name);
    startTimes.erase(name);

    return duration_cast<Period>(stopTime - startTime);
  }

private:
  std::unordered_map<std::string, Time> startTimes;
};
