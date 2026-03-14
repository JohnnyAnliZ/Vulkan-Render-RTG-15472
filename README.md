# Vulkan renderer
My work on top of nakluV https://github.com/15-472/nakluV, the Vulkan tutorial for the class 15472 at CMU

# s72 Loader, s72 format to C++ data

this is how to load a scene:

go to the nakluV directory：
cd C:\Users\zal17\Documents\RealTimeG\nakluV

run viewer with a scene:
bin\main --scene scenes\examples\

tonemap specifiers:
bin\main --scene scenes\examples\materials.s72 --exposure 1 --tone-map REINHARD

# cubemap utility
supports ggx or lambertian convolving of cubemap environment images
bin\cube scenes\examples\ox_bridge_morning.png --lambertian scenes\examples\ox_bridge_morning_mine.png

