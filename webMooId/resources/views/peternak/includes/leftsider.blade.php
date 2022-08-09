<nav class="pcoded-navbar">
    <!-- SIDEBAR TOGGLE -->
    <div class="sidebar_toggle"><a href="#"><i class="icon-close icons"></i></a></div>
    <!-- SIDEBAR MENU -->
    <div class="pcoded-inner-navbar main-menu">
        <div class="">
            <div class="main-menu-header" style="background: url('{{ asset('assets/images/user-bg.png') }}')">
                <img class="img-80 img-radius" src="{{ (@$auth->avatar) ? storage_path('public/peternakan/' . $auth->avatar) : asset('assets/images/avatar-0.png') }}"
                    alt="User-Profile-Image">
                <div class="user-details">
                    <span id="more-details">{{ (@$auth->username) ? $auth->username : 'Guest' }}</span>
                </div>
            </div>
        </div>
        <div class="pcoded-navigation-label">Navigasi</div>
        <ul class="pcoded-item pcoded-left-item">
            <li {{ @$activeMenu == 'beranda' ? 'class=active' : '' }}>
                <a href="{{ route('beranda') }}" class="waves-effect waves-dark">
                    <span class="pcoded-micon"><i class="ti-home"></i><b>N</b></span>
                    <span class="pcoded-mtext">Beranda</span>
                    <span class="pcoded-mcaret"></span>
                </a>
            </li>
            <li {{ @$activeMenu == 'tempat' ? 'class=active' : '' }}>
                <a href="{{ route('tempat.index') }}" class="waves-effect waves-dark">
                    <span class="pcoded-micon"><i class="ti-direction"></i><b>N</b></span>
                    <span class="pcoded-mtext">Tempat</span>
                    <span class="pcoded-mcaret"></span>
                </a>
            </li>
            <li {{ @$activeMenu == 'sapi' ? 'class=active' : '' }}>
                <a href="{{ route('sapi.index') }}" class="waves-effect waves-dark">
                    <span class="pcoded-micon"><i class="ti-heart-broken"></i><b>N</b></span>
                    <span class="pcoded-mtext">Sapi</span>
                    <span class="pcoded-mcaret"></span>
                </a>
            </li>
            <li {{ @$activeMenu == 'dokter' ? 'class=active' : '' }}>
                <a href="{{ route('dokter.index') }}" class="waves-effect waves-dark">
                    <span class="pcoded-micon"><i class="ti-id-badge"></i><b>N</b></span>
                    <span class="pcoded-mtext">Dokter</span>
                    <span class="pcoded-mcaret"></span>
                </a>
            </li>
            <li {{ @$activeMenu == 'obat' ? 'class=active' : '' }}>
                <a href="{{ route('obat.index') }}" class="waves-effect waves-dark">
                    <span class="pcoded-micon"><i class="ti-receipt"></i><b>N</b></span>
                    <span class="pcoded-mtext">Obat</span>
                    <span class="pcoded-mcaret"></span>
                </a>
            </li>
            <!--
            <li>
                <a href="#" class="waves-effect waves-dark">
                    <span class="pcoded-micon"><i class="ti-notepad"></i><b>N</b></span>
                    <span class="pcoded-mtext">History</span>
                    <span class="pcoded-mcaret"></span>
                </a>
            </li>
            <li>
                <a href="#" class="waves-effect waves-dark">
                    <span class="pcoded-micon"><i class="ti-write"></i><b>N</b></span>
                    <span class="pcoded-mtext">Artikel</span>
                    <span class="pcoded-mcaret"></span>
                </a>
            </li>
            <li>
                <a href="#" class="waves-effect waves-dark">
                    <span class="pcoded-micon"><i class="ti-write"></i><b>N</b></span>
                    <span class="pcoded-mtext">Petugas</span>
                    <span class="pcoded-mcaret"></span>
                </a>
            </li>
            -->
        </ul>
    </div>
</nav>
