#import <AppKit/AppKit.h>
#import <CoreAudio/CoreAudio.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
static OSStatus changed(AudioObjectID o,UInt32 n,const AudioObjectPropertyAddress *a,void *ctx){return noErr;}
// Follow macOS output changes for this bottle; only notify, never close the game.
int main(int argc,char **argv){if(argc!=3)return 2;
 int lock=open(argv[2],O_CREAT|O_RDWR,0600);if(lock<0||flock(lock,LOCK_EX|LOCK_NB)){fprintf(stderr,"audio watcher lock: %s\n",strerror(errno));return 0;}
 unsetenv("DYLD_INSERT_LIBRARIES");
 AudioObjectPropertyAddress watch={kAudioHardwarePropertyDefaultOutputDevice,kAudioObjectPropertyScopeGlobal,kAudioObjectPropertyElementMain};AudioObjectAddPropertyListener(kAudioObjectSystemObject,&watch,changed,NULL);fprintf(stderr,"audio watcher started\n");AudioDeviceID last=0;BOOL seen=NO;unsigned absent=0;
 for(unsigned seconds=0;seconds<86400;seconds++){@autoreleasepool{
  BOOL running=NO;for(NSRunningApplication *app in NSWorkspace.sharedWorkspace.runningApplications){NSString *p=app.executableURL.path;if([p containsString:@"/winetemp-"]&&[p.lastPathComponent isEqualToString:@"AION2"]){running=YES;break;}}
  if(running){seen=YES;absent=0;}else{absent++;if((seen&&absent>=3)||(!seen&&seconds>=300)){fprintf(stderr,"audio watcher game absent; exiting\n");break;}}
  AudioObjectPropertyAddress a={kAudioHardwarePropertyDefaultOutputDevice,kAudioObjectPropertyScopeGlobal,kAudioObjectPropertyElementMain};AudioDeviceID current=0;UInt32 size=sizeof(current);
  if(!AudioObjectGetPropertyData(kAudioObjectSystemObject,&a,0,NULL,&size,&current)&&current){
   if(!last)last=current;
   else if(current!=last&&running){NSTask *task=[[NSTask alloc]init];task.launchPath=@"/bin/zsh";task.arguments=@[[NSString stringWithUTF8String:argv[1]]];task.standardOutput=[NSFileHandle fileHandleWithStandardError];task.standardError=[NSFileHandle fileHandleWithStandardError];NSError *error=nil;if([task launchAndReturnError:&error]){[task waitUntilExit];if(task.terminationStatus==0)last=current;}}
  }
 }[[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:1.0]];}
 close(lock);return 0;}
