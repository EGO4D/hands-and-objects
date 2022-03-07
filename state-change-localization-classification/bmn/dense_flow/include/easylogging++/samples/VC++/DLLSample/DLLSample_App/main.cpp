#include <iostream>

#include "../DLLSample_Lib/MyMaths.h"

// There are two ways to share repository
// one way is:
// SHARE_EASYLOGGINGPP(sharedLoggingRepository())

// Other way is
// INITIALIZE_NULL_EASYLOGGINGPP
// and in main function:
// int main(int argc, char** argv) {
//     el::Helpers::setStorage(sharedLoggingRepository());
//     ...
//     START_EASYLOGGINGPP(argc, argv);
//     ...
// }

INITIALIZE_NULL_EASYLOGGINGPP

int main() {
	el::Helpers::setStorage(sharedLoggingRepository());
	Math::MyMaths::logAdd(1, 2);
    LOG(INFO) << "Wow";
	el::Loggers::reconfigureAllLoggers(el::Level::Global, el::ConfigurationType::Format, "%datetime %msg");
	Math::MyMaths::logAdd(1, 2);
	LOG(INFO) << "This is after reconfiguration from main";
	system("pause");
	return 0;
}