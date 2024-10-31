import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import geopandas as gpd
from shapely.geometry import Point

# 初始化参数 - 使用整数计算
grid_size_LON = 2000
grid_size_LAT =1000
initial_population = 200
growth_rate_int = 5  # 5% 转换为整数表示
split_population = 400
num_tribes = 2

# 预计算网格转换因子，避免重复计算
LON_TO_X = grid_size_LON / 360
LAT_TO_Y = grid_size_LAT / 180

# 加载世界地图数据
world_path = 'ne_110m_admin_0_countries.shp'
world = gpd.read_file(world_path)

bounds = world.total_bounds
minx, miny, maxx, maxy = bounds
print(f"地图的边界范围: ({minx}, {miny}) 到 ({maxx}, {maxy})")

# 初始化网格和部落
grid = np.zeros((grid_size_LON, grid_size_LAT), dtype=np.int32)  # 使用整数类型
tribes = []
tribe_markers = []  # 存储部落标记

# 警告状态
warning_active = False
warning_alpha = 1.0

def convert_to_grid(lon, lat):
    x = int((lon + 180) * LON_TO_X)
    y = int((-lat + 90) * LAT_TO_Y)
    return x, y

def convert_to_lonlat(x, y):
    lon = int(x / LON_TO_X - 180)
    lat = int(-(y / LAT_TO_Y - 90))
    return lon, lat

def initialize_first_tribe():
    global warning_active
    chosen_lon = 30
    chosen_lat = 0
    x, y = convert_to_grid(chosen_lon, chosen_lat)
    
    if not is_land(x, y):
        warning_active = True
        return False
    
    grid[x, y] = initial_population
    tribes.append((x, y, initial_population, False))
    return True

def move_tribe(x, y):
    move_steps = random.randint(0, 3)
    if move_steps == 0:
        return x, y
    
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), 
                 (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    random.shuffle(directions)
    
    tribe_population = grid[x, y]
    
    for dx, dy in directions:
        new_x = x + dx * move_steps
        new_y = y + dy * move_steps
        
        if (0 <= new_x < grid_size_LON and 
            0 <= new_y < grid_size_LAT and 
            grid[new_x, new_y] == 0 and 
            is_land(new_x, new_y)):
            
            # 计算水域格子数量
            water_cells = 0
            for step in range(1, move_steps + 1):
                check_x = x + dx * step
                check_y = y + dy * step
                if not is_land(check_x, check_y):
                    water_cells += 1
            
            # 根据水域距离计算所需人口
            if water_cells > 0:
                # 每格水域需要50人口支持
                required_population = water_cells * 50
                
                # 考虑移动距离的额外消耗
                distance_factor = move_steps * 10
                total_required = required_population + distance_factor
                
                if tribe_population > total_required:
                    # 成功跨海后减少人口
                    grid[x, y] = tribe_population - total_required
                    return new_x, new_y
            else:
                # 陆地移动不需要额外条件
                return new_x, new_y
                
    return x, y

def is_land(x, y):
    lon, lat = convert_to_lonlat(x, y)
    point = Point(lon, lat)
    try:
        return world.contains(point).any()
    except:
        return False

def update(frame):
    global tribes, grid, year, warning_active, warning_alpha, tribe_markers, tribe_contacts
    
    if warning_active:
        warning_alpha = 0.5 + 0.5 * np.sin(frame * 0.5)
        warning_patch.set_alpha(warning_alpha)
        warning_text.set_alpha(warning_alpha)
        return [im, warning_patch, warning_text, year_text] + tribe_markers

    year += 1
    
    # 清除旧的标记
    for marker in tribe_markers:
        marker.remove()
    tribe_markers.clear()
    
    # 如果没有部落，直接返回
    if not tribes:
        im.set_array(grid)
        year_text.set_text(f'Year: {year} | Tribes: 0')
        return [im, warning_patch, warning_text, year_text]
    
    # 每20年随机消除50%的部落
    if year % 20 == 0:
        num_tribes_to_keep = len(tribes) // 2
        surviving_indices = random.sample(range(len(tribes)), num_tribes_to_keep)
        tribes = [tribes[i] for i in surviving_indices]
        tribe_contacts.clear()
        if not tribes:  # 如果所有部落都被清除
            grid.fill(0)
            im.set_array(grid)
            year_text.set_text(f'Year: {year} | Tribes: 0')
            return [im, warning_patch, warning_text, year_text]
    
    # 将部落位置转换为numpy数组以加速计算
    tribe_positions = np.array([(x, y) for x, y, _, _ in tribes])
    
    # 使用numpy计算部落间的距离矩阵
    X = tribe_positions[:, np.newaxis, :]
    Y = tribe_positions[np.newaxis, :, :]
    distances = np.sqrt(((X - Y) ** 2).sum(axis=2))
    
    # 找出相邻的部落（距离小于等于√2的部落对）
    adjacent_pairs = np.where((distances <= np.sqrt(2)) & (distances > 0))
    
    # 更新接触时间和处理需要移除的部落
    tribes_to_remove = set()
    new_contacts = {}
    
    for i, j in zip(*adjacent_pairs):
        if i < j:  # 只处理一次每对部落
            contact_key = (i, j)
            if contact_key in tribe_contacts:
                tribe_contacts[contact_key] += 1
                if tribe_contacts[contact_key] >= 3:
                    tribes_to_remove.add(random.choice([i, j]))
            else:
                new_contacts[contact_key] = 1
    
    # 更新网格
    grid.fill(0)
    for i, (x, y, pop, _) in enumerate(tribes):
        if not is_land(x, y):
            continue
        grid[x, y] = pop
        
        # 计算四邻域的部落数量
        neighbor_count = 0
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < grid_size_LON and 
                0 <= ny < grid_size_LAT and 
                grid[nx, ny] > 0):
                neighbor_count += 1
        
        if neighbor_count >= 3:
            tribes_to_remove.add(i)
    
    # 更新接触记录
    tribe_contacts = new_contacts
    
    # 移除标记的部落
    if tribes_to_remove:
        tribes = [tribe for i, tribe in enumerate(tribes) if i not in tribes_to_remove]
    
    # 处理剩余部落的移动和分裂
    new_tribes = []
    grid.fill(0)
    
    for idx, (x, y, population, stopped) in enumerate(tribes):
        if not stopped and is_land(x, y):
            # 人口增长
            population = population + (population * growth_rate_int) // 100
            
            # 处理分裂
            if population >= split_population:
                new_x, new_y = move_tribe(x, y)
                if is_land(new_x, new_y):
                    new_tribes.append((new_x, new_y, population // 2, False))
                    population //= 2
            
            # 移动部落
            new_x, new_y = move_tribe(x, y)
            tribes[idx] = (new_x, new_y, population, stopped)
            grid[new_x, new_y] = population
            
            # 添加标记
            lon, lat = convert_to_lonlat(new_x, new_y)
            tribe_markers.append(ax.plot(lon, lat, 'r.', markersize=2)[0])
    
    tribes.extend(new_tribes)

    im.set_array(grid)
    year_text.set_text(f'Year: {year} | Tribes: {len(tribes)}')
    return [im, warning_patch, warning_text, year_text] + tribe_markers
# 创建图形
fig, ax = plt.subplots(figsize=(12, 8))
world.boundary.plot(ax=ax, color='black', linewidth=0.5)
im = ax.imshow(grid, cmap='YlGn', interpolation='nearest', extent=[-180, 180, -90, 90], alpha=0.7)
plt.grid(True)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
year_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='red', fontsize=12)

ax.set_xlim([-180, 180])
ax.set_ylim([-90, 90])

# 警告显示
warning_patch = plt.Rectangle((0.7, 0.8), 0.25, 0.15,
                            transform=ax.transAxes,
                            facecolor='red',
                            alpha=0,
                            zorder=1000)
ax.add_patch(warning_patch)

warning_text = ax.text(0.825, 0.875,
                      'Invalid Start\nPoint!',
                      transform=ax.transAxes,
                      color='white',
                      horizontalalignment='center',
                      verticalalignment='center',
                      fontsize=10,
                      zorder=1001)
warning_text.set_visible(False)

year = 0
if initialize_first_tribe():
    warning_text.set_visible(False)
    warning_patch.set_alpha(0)
else:
    warning_text.set_visible(True)
    warning_patch.set_alpha(1)

ani = FuncAnimation(fig, update, interval=50, blit=True)
plt.show()
