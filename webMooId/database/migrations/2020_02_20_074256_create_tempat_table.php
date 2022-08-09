<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateTempatTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('tempat', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->unsignedBigInteger('id_peternakan');
            $table->enum('jenis_tempat', ['tertutup','terbuka']);
            $table->integer('kapasitas_sapi');
            $table->foreign('id_peternakan')->references('id')->on('peternakan')->onUpdate('cascade')->onDelete('cascade');
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('tempat');
    }
}
