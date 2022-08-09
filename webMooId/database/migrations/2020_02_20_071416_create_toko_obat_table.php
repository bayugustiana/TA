<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateTokoObatTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('toko_obat', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->string('nama_toko_obat');
            $table->string('nama_lengkap');
            $table->string('email');
            $table->string('password');
            $table->string('no_telepon');
            $table->string('avatar');
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('toko_obat');
    }
}
