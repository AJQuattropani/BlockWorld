#include "Application.h"

#ifndef DIST
int main(int argc, char* argv[]) {
	Application application;
	return application.run();
}
#else
#include "Windows.h"
int WinMain(HINSTANCE hInstances, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {

	Application application;
	return application.run();
}


#endif